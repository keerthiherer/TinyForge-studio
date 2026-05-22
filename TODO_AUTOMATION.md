- [ ] Implement COCO-backed annotation storage + CRUD + YOLO export endpoints in web_app.py
- [ ] Refactor object_detection_workflow.py::prepare_yolo_dataset() to export YOLO TXT from COCO bbox annotations (coco_store/<project>/coco_train.json and coco_test.json)
- [ ] Ensure class name ↔ category_id ↔ YOLO class index mapping consistent
- [ ] Update web_app.py YOLO training pipeline to use COCO-backed exporter
- [ ] Remove any fake boxes generation; consume only real COCO bbox annotations
- [ ] Implement auto-label pipeline APIs (POST /api/auto-label/*) and frontend wiring if required
- [ ] Smoke test: run a quick export and verify yolo_auto labels

