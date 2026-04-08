"""
collect_data.py
Record your own training samples from webcam.

Usage:
    python collect_data.py --class attentive  --n 150
    python collect_data.py --class distracted --n 150
    python collect_data.py --class disengaged --n 150

Controls:  SPACE = capture frame  |  Q = quit
"""

import argparse
import os

import cv2

CLASSES = ["attentive", "distracted", "disengaged"]

parser = argparse.ArgumentParser()
parser.add_argument("--class", dest="cls", required=True, choices=CLASSES)
parser.add_argument("--n",     type=int,   default=150)
parser.add_argument("--cam",   type=int,   default=0)
args = parser.parse_args()

save_dir = os.path.join("data", args.cls)
os.makedirs(save_dir, exist_ok=True)

existing = len(os.listdir(save_dir))
cap      = cv2.VideoCapture(args.cam)
count    = 0

print(f"\nRecording class: '{args.cls}'  (target: {args.n} samples)")
print("SPACE = capture  |  Q = quit\n")

while count < args.n:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    cv2.putText(frame, f"{args.cls.upper()}  {count}/{args.n}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "SPACE=capture  Q=quit",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        fname = os.path.join(save_dir, f"{args.cls}_{existing + count:04d}.jpg")
        cv2.imwrite(fname, frame)
        count += 1
        print(f"  Saved {count}/{args.n}")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nDone! {count} images saved to data/{args.cls}/")
