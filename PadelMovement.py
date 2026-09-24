import cv2
import csv
from ultralytics import YOLO
from ultralytics import solutions
from ultralytics.utils.downloads import safe_download
import pandas as pd

model = YOLO("yolo26n-pose.pt")
# results = model.predict(source=r"C:\Padel_Project\video1.mp4", show = True, save = True)
results = model.track(source=r"C:\Padel_Project\video1.mp4", show = True, save = True, project=r"C:\Padel_Project\results",
    name="tracking_lebron", persist= True,  imgsz = 1920, conf = 0.05, tracker= r"C:\Padel_Project\custom_botsort.yaml")

cv2.destroyAllWindows()


for result in results:
    xy = result.keypoints.xy
    xyn = result.keypoints.xyn
    kpts = result.keypoints.data



with open(r"C:\Padel_Project\results\keypoints.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "person_idx", "keypoint_id", "x", "y", "conf"])

    for frame_idx, result in enumerate(results):
        xy = result.keypoints.xy
        xyn = result.keypoints.xyn
        kpts = result.keypoints.data  # (num_people, num_keypoints, 3) -> x, y, conf

        for person_idx, person_kpts in enumerate(kpts):
            for kp_idx, (x, y, conf) in enumerate(person_kpts.tolist()):
                writer.writerow([frame_idx, person_idx, kp_idx, x, y, conf])

with open(r"C:\Padel_Project\results\keypoints.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "track_id", "person_idx", "keypoint_id", "x", "y", "conf"])

    for frame_idx, result in enumerate(results):
        if result.keypoints is None:
            continue

        kpts = result.keypoints.data  # (num_people, num_keypoints, 3) -> x, y, conf

        # track_id per detected person this frame (aligned by detection order)
        track_ids = result.boxes.id
        track_ids = track_ids.int().tolist() if track_ids is not None else [None] * len(kpts)

        for person_idx, person_kpts in enumerate(kpts):
            tid = track_ids[person_idx] if person_idx < len(track_ids) else None

            for kp_idx, (x, y, conf) in enumerate(person_kpts.tolist()):
                writer.writerow([frame_idx, tid, person_idx, kp_idx, x, y, conf])

print("Saved keypoints.csv with track_id")

id_map = {1: 1, 232: 1, 233: 1}
df["player_id"] = df["track_id"].map(id_map)

player_df = df.dropna(subset=["player_id"]).copy()
player_df["player_id"] = player_df["player_id"].astype(int)

player_df.to_csv(r"C:\Padel_Project\results\player1_keypoints.csv", index=False)
print(player_df.frame.min(), "to", player_df.frame.max(), "-", player_df.frame.nunique(), "frames covered")

id_map = {
    1: 1, 232: 1, 233: 1,   # player 1, re-identified twice near the end
    2: 2,
    3: 3,
    4: 4,
}

players_df = df[df.track_id.isin(id_map.keys())].copy()
players_df["player_id"] = players_df["track_id"].map(id_map)

players_df.to_csv(r"C:\Padel_Project\results\players_keypoints.csv", index=False)
    
df = pd.read_csv(r"C:\Padel_Project\results\players_keypoints.csv")
print(df.head())
