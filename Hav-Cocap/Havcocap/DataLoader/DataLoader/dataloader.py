import av
import numpy as np
import collections

class GOPDataloader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.container = None
        
    def process(self):
        """
        Generates GOPs from the video file.
        Each yielded item is a dict containing:
        - gop_index: Index of the GOP
        - start_time: Start timestamp of the GOP (seconds)
        - end_time: End timestamp of the GOP (seconds)
        - video_frames: List of av.VideoFrame objects
        - audio_samples: numpy array of audio samples (stereo/mono based on source)
        - sample_rate: Audio sample rate
        """
        try:
            self.container = av.open(self.video_path)
            
            # Setup streams
            video_stream = self.container.streams.video[0]
            audio_stream = self.container.streams.audio[0] if self.container.streams.audio else None
            
            # Buffers
            current_video_frames = []
            audio_buffer = collections.deque() # Store (pts_seconds, samples_array, sample_rate)
            
            gop_start_time = None
            gop_index = 0
            
            # Iterate through all packets
            for packet in self.container.demux(video=0, audio=0) if audio_stream else self.container.demux(video=0):
                if packet.dts is None:
                    continue
                    
                if packet.stream.type == 'video':
                    # Decode framing
                    for frame in packet.decode():
                        # We use packet.is_keyframe check on the PACKET side usually, but decoding gives frames.
                        # av.VideoFrame.key_frame exists.
                        
                        is_iframe = frame.key_frame
                        timestamp = frame.time
                        
                        if is_iframe:
                            # If we have a previous GOP accumulating, verify and yield it
                            if gop_start_time is not None:
                                gop_end_time = timestamp
                                
                                # Process Audio for this GOP
                                segment_audio, sr = self._extract_audio_segment(
                                    audio_buffer, gop_start_time, gop_end_time
                                )
                                
                                yield {
                                    'gop_index': gop_index,
                                    'start_time': gop_start_time,
                                    'end_time': gop_end_time,
                                    'video_frames': current_video_frames,
                                    'audio_samples': segment_audio,
                                    'sample_rate': sr
                                }
                                
                                gop_index += 1
                                current_video_frames = []
                                # Clean up old audio from buffer to save memory
                                self._clean_audio_buffer(audio_buffer, gop_end_time)
                                
                            gop_start_time = timestamp
                        
                        current_video_frames.append(frame)
                        
                elif packet.stream.type == 'audio':
                    for frame in packet.decode():
                        # Store audio frames
                        # frame.time is the timestamp of the first sample in the frame
                        data = frame.to_ndarray() # Shape (channels, samples) usually, or (samples, channels) depending on format. PyAV usually planar (channels, samples)
                        # Let's transpose to (samples, channels) standard for processing? Or keep as is.
                        # Standard is usually (samples, channels) for most ML libs, but PyAV gives (channels, samples) for planar.
                        # We'll check format. assuming numpy.
                        
                        # frame.rate is sample rate
                        audio_buffer.append({
                            'time': frame.time,
                            'data': data,
                            'rate': frame.rate,
                            'samples': frame.samples
                        })
            
            # Final GOP (if any frames left)
            if current_video_frames and gop_start_time is not None:
                 # End of video is end time
                 # Estimate end time based on last frame duration? 
                 # Or just use last frame time.
                 last_frame_time = current_video_frames[-1].time
                 # Usually GOP length is consistent, but for the last one we just take what we have.
                 # Let's use the last frame's time + its duration if possible, or just last frame time.
                 # But we don't have next I-frame time.
                 # We can use container duration or just last frame time.
                 
                 segment_audio, sr = self._extract_audio_segment(
                     audio_buffer, gop_start_time, last_frame_time
                 )
                 
                 yield {
                    'gop_index': gop_index,
                    'start_time': gop_start_time,
                    'end_time': last_frame_time,
                    'video_frames': current_video_frames,
                    'audio_samples': segment_audio,
                    'sample_rate': sr
                }

        except Exception as e:
            print(f"Error processing video: {e}")
            raise
        finally:
            if self.container:
                self.container.close()

    def _extract_audio_segment(self, audio_buffer, start_time, end_time):
        """
        Extracts and concatenates audio samples relevant to the [start_time, end_time] range.
        Returns: (concatenated_samples, sample_rate)
        """
        if not audio_buffer:
            return None, None
            
        relevant_data = []
        sample_rate = audio_buffer[0]['rate']
        
        # Audio buffer contains decoded blocks.
        # We assume they are roughly in order.
        
        # Simple algorithm: 
        # 1. Collect all frames that overlap with time range.
        # 2. Concatenate.
        # 3. Trim edges based on exact sample calculations.
        
        collected_frames = []
        
        for item in audio_buffer:
            item_start = item['time']
            duration = item['samples'] / item['rate']
            item_end = item_start + duration
            
            # Check overlap
            if item_end > start_time and item_start < end_time:
                collected_frames.append(item)
        
        if not collected_frames:
            return np.array([]), sample_rate
            
        # Concatenate
        # Assumes consistent channels and rate
        # item['data'] is (channels, samples) for planar formats which PyAV often converts to
        # Let's verify shape. 
        # We will assume we want to stack along the time axis (last axis for planar or first for packed)
        # PyAV to_ndarray typically returns (n_channels, n_samples) for planar formats (fltp) 
        # and (n_samples, n_channels) for packed specific formats?
        # Actually PyAV `to_ndarray` behavior depends on format.
        # We will inspect first frame.
        
        first_shape = collected_frames[0]['data'].shape
        # Heuristic: audio usually has more samples than channels.
        if first_shape[0] < 10 and len(first_shape) > 1:
             # Likely (channels, samples)
             axis = 1
             is_planar = True
        else:
             axis = 0
             is_planar = False
             
        full_arr = np.concatenate([x['data'] for x in collected_frames], axis=axis)
        
        # Now trim
        # Start time of the first collected frame
        first_frame_start = collected_frames[0]['time']
        
        # Calculate offset in seconds
        start_offset_sec = start_time - first_frame_start
        # If start_offset_sec < 0, it means the requested start is BEFORE the first audio frame we have.
        # We just take from the beginning (index 0).
        start_offset_sec = max(0, start_offset_sec)
        
        end_offset_sec = end_time - first_frame_start
        
        start_idx = int(start_offset_sec * sample_rate)
        end_idx = int(end_offset_sec * sample_rate)
        
        if is_planar:
            # (channels, samples)
            # Ensure indices are within bounds
            start_idx = min(start_idx, full_arr.shape[1])
            end_idx = min(end_idx, full_arr.shape[1])
            sliced = full_arr[:, start_idx:end_idx]
        else:
            # (samples, channels)
            start_idx = min(start_idx, full_arr.shape[0])
            end_idx = min(end_idx, full_arr.shape[0])
            sliced = full_arr[start_idx:end_idx, :]
            
        return sliced, sample_rate

    def _clean_audio_buffer(self, audio_buffer, retention_time_boundary):
        """
        Removes audio frames that are entirely before the retention time.
        """
        while audio_buffer:
            item = audio_buffer[0]
            item_start = item['time']
            duration = item['samples'] / item['rate']
            item_end = item_start + duration
            
            # If this frame ends before our new boundary (minus some safety margin), drop it
            if item_end < retention_time_boundary:
                audio_buffer.popleft()
            else:
                break
