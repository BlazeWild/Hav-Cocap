import json
import random

def extract():
    with open('HavCocap_new/logs/charades_captioning/caption_greedy_pred_validation_2026-03-19T12:35:09.json', 'r') as f:
        data = json.load(f)

    all_captions = []
    for vid_id, captions in data['results'].items():
        for cap in captions:
            # We can use simple heuristics or ROUGE-L/CIDEr via string matching to guess "good" vs "bad"
            # Here we just use a basic string similarity proxy (Jaccard-like over words)
            pred_words = set(cap['sentence'].lower().split())
            gt_words = set(cap['gt_sentence'].lower().split())
            intersection = len(pred_words.intersection(gt_words))
            union = len(pred_words.union(gt_words))
            score = intersection / union if union > 0 else 0
            
            all_captions.append({
                'video_id': vid_id,
                'sentence': cap['sentence'],
                'gt_sentence': cap['gt_sentence'],
                'score': score
            })

    # Sort to get good and bad
    all_captions.sort(key=lambda x: x['score'])
    
    # Worst 100 to pick 20 random from
    worst = all_captions[:100]
    
    # Best 100 to pick 20 random from
    best = all_captions[-100:]
    
    if len(worst) >= 20: bad_samples = random.sample(worst, 20)
    else: bad_samples = worst
        
    if len(best) >= 20: good_samples = random.sample(best, 20)
    else: good_samples = best

    output = {
        'good_captions': [{"video_id": s['video_id'], "sentence": s['sentence'], "gt_sentence": s['gt_sentence']} for s in good_samples],
        'bad_captions': [{"video_id": s['video_id'], "sentence": s['sentence'], "gt_sentence": s['gt_sentence']} for s in bad_samples]
    }

    with open('selected_captions.json', 'w') as f:
        json.dump(output, f, indent=4)
        
    print("Wrote 20 good and 20 bad captions to selected_captions.json")

if __name__ == '__main__':
    extract()
