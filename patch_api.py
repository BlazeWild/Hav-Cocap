import re

file_path = "./CoCap_edited/Compressed-Video-Reader/src/cv_reader/api.cpp"
with open(file_path, "r") as f:
    code = f.read()

old_block = """                uint8_t *res_data[1] = {(uint8_t *) residual_arr->data};
                int dst_stride[1] = {width * 3};

                sws_scale(sws_ctx, frame->data, frame->linesize, 0, dec_ctx->height, res_data, dst_stride);"""

new_block = """                int stride = (width * 3 + 31) & ~31;
                uint8_t *tmp_buffer = new uint8_t[height * stride + 64]();
                uint8_t *res_data[1] = {tmp_buffer};
                int dst_stride[1] = {stride};
                sws_scale(sws_ctx, frame->data, frame->linesize, 0, dec_ctx->height, res_data, dst_stride);
                for (int row = 0; row < height; ++row) {
                    memcpy((uint8_t *)residual_arr->data + row * width * 3, tmp_buffer + row * stride, width * 3);
                }
                delete[] tmp_buffer;"""

if old_block in code:
    code = code.replace(old_block, new_block)
    with open(file_path, "w") as f:
        f.write(code)
    print("Patch successful.")
else:
    print("Old block not found!")
