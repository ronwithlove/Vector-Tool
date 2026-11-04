"""命令行：python cli.py 输入图片 [-o 输出.svg]"""
import argparse, json, time
from vectortool.pipeline import Params, decode_image, segment
from vectortool.trace import vectorize_segmentation

ap = argparse.ArgumentParser()
ap.add_argument("input"); ap.add_argument("-o", "--output")
ap.add_argument("--params", default="{}", help='JSON，例如 {"max_colors": 8}')
a = ap.parse_args()
p = Params.from_dict(json.loads(a.params))
t = time.time()
img = decode_image(open(a.input, "rb").read())
seg = segment(img, p)
svg, shapes, stats = vectorize_segmentation(seg, p)
stats["total_ms"] = int((time.time() - t) * 1000)
out = a.output or a.input.rsplit(".", 1)[0] + ".svg"
open(out, "w").write(svg)
print(json.dumps(stats, ensure_ascii=False), "palette:", seg.colors_hex)
