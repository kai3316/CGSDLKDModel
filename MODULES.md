# Modules

Maps each component in the paper to its implementation in the code.

## The four modules

The four modules replace parts of the YOLO11 pipeline. They are registered in ultralytics/nn/tasks.py through `from ultralytics.nn.extra_modules import *`.

### Gr-CSP (RGCSPELAN)

File: ultralytics/nn/extra_modules/block.py

Backbone feature-extraction block. A reparameterizable 3x3 convolution (RepConv) at the entry, then the input is split into two channel groups (cross-stage partial) and refined with a series of 3x3 convolutions. Used in configs/yolo11-student.yaml (backbone and head).

### Dy-Sample (DySample)

File: ultralytics/nn/extra_modules/block.py

Content-adaptive upsampling. Predicts a per-location sampling offset and samples with grid_sample, replacing the fixed nearest/bilinear upsample in the neck. Used in the head of the student config as `DySample, [2, 'lp']`.

### Sf-Conv (FeaturePyramidSharedConv)

File: ultralytics/nn/extra_modules/block.py

Shared-weight multi-scale fusion. Parallel dilated convolutions (rates 1, 3, 5) that share the same convolution weight. Used in the student config as `FeaturePyramidSharedConv, [1024]` (layer 9).

### Led-Head (Detect_Efficient)

File: ultralytics/nn/extra_modules/head.py

Lightweight decoupled detection head. For each scale, a stem of two 3x3 group convolutions (groups = channels / 16), then separate box and class branches. The box branch outputs reg_max x 4 and is decoded with a DFL module; the class branch outputs nc. Used as the final `Detect_Efficient, [nc]` head.

## Knowledge distillation

### Feature-level distillation (CWDLoss)

File: ultralytics/utils/distill_loss.py

Channel-wise distillation. Each channel map is turned into a spatial probability distribution (softmax over HxW positions with temperature tau), and the student's distribution is pulled toward the teacher's with a channel-wise KL divergence (teacher as reference): L_CWD = tau^2 * sum_i KL(p_i^T || p_i^S).

### Output-level distillation (LogicalLoss, OutputLoss)

File: ultralytics/utils/distill_loss.py

DetectionDistiller creates LogicalLoss with logical_loss_type = "l2", which selects OutputLoss. The prediction tensor is split into box regression (reg_max x 4, the DFL distribution) and classification (nc). Each anchor is weighted by the teacher confidence W = max_c sigmoid(T_scores), and a confidence-weighted MSE (L2) is minimized on both branches.

### Training loop (DetectionDistiller)

File: ultralytics/models/yolo/detect/distill.py

Extends BaseTrainer. Main options (set in distill.py):

- kd_loss_type: "feature", "logical", or "all"
- feature_loss_type: "cwd" (used), "mgd", "mimic", "chsim", "sp"
- logical_loss_type: "l2"
- teacher_kd_layers / student_kd_layers: comma-separated layer indices, e.g. "12,15,18,21"

## Model configs

| Config file | Meaning | Params |
|-------------|---------|--------|
| yolo11.yaml | YOLOv11n baseline | 2.582M |
| yolo11-student.yaml | GSDL-YOLOv11n | 2.132M |
| yolo11-studentCCC.yaml | compressed student CGSDL-YOLOv11n | 0.578M |
| yolo11-techer.yaml | GSDL-YOLOv11m teacher | 17.185M |
| yolo11-djz2.yaml | teacher variant | - |
| yolo11-D+L.yaml | ablation: Dy-Sample + Led-Head | - |
| yolo11-D+s.yaml | ablation: Dy-Sample + standard head | - |
| yolo11-G+L.yaml | ablation: Gr-CSP + Led-Head | - |
| yolo11-CSP-PMSFA--RGCSPELAN.yaml | earlier experiment | - |

## Dependencies

The core path needs only torch, timm, einops, and the standard Ultralytics dependencies. The fork also contains optional third-party operators (Mamba, DCNv3/DCNv4, selective-scan, KAN-conv) that are not used by this model and are imported with try/except.
