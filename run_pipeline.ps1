$ErrorActionPreference = "Continue"
Set-Location "c:\Users\yangn\Downloads\Task Based Retinal Prosthesis Simulation"
$py = ".\venv\Scripts\python.exe"

Write-Output "=== STAGE 1: CNN TRAINING START $(Get-Date -Format o) ==="
& $py -m end2end.train_moving_mnist --epochs 10 --batch 64 --sim-hw 128 --backbone cnn --topk-frac 0.0556 --out runs/e2e_cnn_full 2>&1 | Tee-Object -FilePath train_cnn.log
Write-Output "STAGE1_CNN_TRAIN_DONE_EXIT_$LASTEXITCODE $(Get-Date -Format o)"

Write-Output "=== STAGE 2: MOE TRAINING START $(Get-Date -Format o) ==="
& $py -m end2end.train_moving_mnist --epochs 10 --batch 64 --sim-hw 128 --moe --moe-experts cnn,mobilenet_v3_small --topk-frac 0.0556 --out runs/e2e_moe_full 2>&1 | Tee-Object -FilePath train_moe.log
Write-Output "STAGE2_MOE_TRAIN_DONE_EXIT_$LASTEXITCODE $(Get-Date -Format o)"

Write-Output "=== STAGE 3: CNN EVAL START $(Get-Date -Format o) ==="
& $py eval_e2e.py --ckpt runs/e2e_cnn_full/ckpt_epoch10.pt --sim-hw 128 --device cuda 2>&1 | Tee-Object -FilePath eval_cnn.log
Write-Output "STAGE3_CNN_EVAL_DONE_EXIT_$LASTEXITCODE $(Get-Date -Format o)"

Write-Output "=== STAGE 4: MOE EVAL START $(Get-Date -Format o) ==="
& $py eval_e2e.py --ckpt runs/e2e_moe_full/ckpt_epoch10.pt --sim-hw 128 --moe --device cuda 2>&1 | Tee-Object -FilePath eval_moe.log
Write-Output "STAGE4_MOE_EVAL_DONE_EXIT_$LASTEXITCODE $(Get-Date -Format o)"

Write-Output "PIPELINE_ALL_DONE $(Get-Date -Format o)"
