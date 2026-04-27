#!/bin/bash
# Reproduce all link prediction experiments (HBNN Table 7 extension)
#
# Usage:
#   bash run.sh              # Run all experiments
#   bash run.sh KleinFC      # Run single model
#   bash run.sh --quick      # Quick test (100 epochs)
#
# Results saved to results/link_prediction/

set -e

DATA="$(cd "$(dirname "$0")" && pwd)/data"
RESULTS="$(cd "$(dirname "$0")" && pwd)/results/link_prediction"
mkdir -p "$RESULTS"

cd "$(dirname "$0")/GyroBN-main"

# --- Config ---
EPOCHS=5000
PRECISION=1  # float64
COMMON="nnet.BN_param.is_bn=False fit.epochs=$EPOCHS fit.double_precision=$PRECISION fit.patience=100 fit.min_epochs=100 fit.log_freq=5000"

if [ "$1" = "--quick" ]; then
    EPOCHS=100
    PRECISION=0
    COMMON="nnet.BN_param.is_bn=False fit.epochs=$EPOCHS fit.double_precision=$PRECISION fit.patience=50 fit.min_epochs=10 fit.log_freq=100"
    shift
fi

# --- Helper ---
run_model() {
    local MODEL=$1 ACT=$2 DATASET=$3 WD=$4
    echo ">>> $MODEL | $DATASET | act=$ACT | wd=$WD"
    python GyroBNH.py \
        dataset.dataset=$DATASET dataset.path=$DATA \
        nnet=$MODEL nnet.model.act=$ACT nnet.optimizer.weight_decay=$WD \
        $COMMON 2>&1 | tee -a "$RESULTS/${MODEL}_${DATASET}.log" | grep "Test set"
}

run_all_datasets() {
    local MODEL=$1 ACT_DISEASE=$2 ACT_AIRPORT=$3 ACT_PUBMED=$4 ACT_CORA=$5

    run_model $MODEL $ACT_DISEASE disease_lp 1e-3
    run_model $MODEL $ACT_AIRPORT airport    1e-3
    run_model $MODEL $ACT_PUBMED  pubmed     1e-3
    run_model $MODEL $ACT_CORA    cora       0
}

# --- Experiments ---
# Format: model, act_disease, act_airport, act_pubmed, act_cora

if [ -z "$1" ] || [ "$1" = "HNN" ]; then
    echo "=== Möbius FC (baseline reproduction) ==="
    run_all_datasets HNN relu relu relu relu
fi

if [ -z "$1" ] || [ "$1" = "KleinFC" ]; then
    echo "=== Klein FC (geodesic, ours) ==="
    run_all_datasets KleinFC relu relu relu relu
fi

if [ -z "$1" ] || [ "$1" = "KleinBFC" ]; then
    echo "=== Klein BFC (Busemann, ours) ==="
    run_all_datasets KleinBFC null tanh null tanh
fi

if [ -z "$1" ] || [ "$1" = "PVFC" ]; then
    echo "=== PV FC (geodesic, ours) ==="
    run_all_datasets PVFC null tanh null tanh
fi

if [ -z "$1" ] || [ "$1" = "PVBFC" ]; then
    echo "=== PV BFC (Busemann, ours) ==="
    run_all_datasets PVBFC null tanh null tanh
fi

if [ -z "$1" ] || [ "$1" = "LorentzFC" ]; then
    echo "=== Lorentz FC (reproduction) ==="
    run_all_datasets LorentzFC relu relu relu relu
fi

echo ""
echo "=== ALL DONE ==="
echo "Results in: $RESULTS/"
