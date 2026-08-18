# Model configs

These YAML files define the model architectures. They are parsed by ultralytics/nn/tasks.py and reference the modules in ../MODULES.md.

| File | Role | Description |
|------|------|-------------|
| yolo11.yaml | baseline | YOLOv11n, 2.582M |
| yolo11-student.yaml | student | GSDL-YOLOv11n, 2.132M |
| yolo11-studentCCC.yaml | student (compressed) | CGSDL-YOLOv11n, 0.578M; the student used in distillation |
| yolo11-techer.yaml | teacher | GSDL-YOLOv11m (uses CSP_PMSFA), 17.185M |
| yolo11-djz2.yaml | teacher variant | used in some teacher runs |
| yolo11-D+L.yaml | ablation | Dy-Sample + Led-Head |
| yolo11-D+s.yaml | ablation | Dy-Sample + standard Detect head |
| yolo11-G+L.yaml | ablation | Gr-CSP + Led-Head |
| yolo11-CSP-PMSFA--RGCSPELAN.yaml | earlier experiment | CSP_PMSFA variant |

Notes:

- The scales.n entry [0.50, 0.25, 1024] controls the nano scale (depth, width, max channels). yolo11-studentCCC.yaml hard-codes the widths to the compressed values.
- nc is overridden at runtime by the names list in the dataset YAML. The 80 in these files is a placeholder.
- Older run logs reference yolo11m-techer.yaml and yolo11m-djz2.yaml, which are not in this snapshot. The equivalent files here are yolo11-techer.yaml and yolo11-djz2.yaml.
