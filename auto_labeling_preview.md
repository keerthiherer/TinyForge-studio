Auto-labeling implementation note

- Backend: add COCO-backed auto-label endpoints + a small in-memory job store for predictions.
- Storage: keep predicted boxes in memory; once approved, merge into coco_store/<project>/coco_<split>.json via existing /api/annotations/save logic.
- Frontend: add “Automatic labeling” button + modal; draw proposed boxes with dashed outlines; approve merges them.

(Actual code changes are in web_app.py / templates/labeling_bbox.html / static/labeling_bbox.js.)

