if [ $# -lt 1 ]; then
    DATA_DIR="./data"
else
    DATA_DIR=$1
fi

EXTRA_ARGS=$2

mkdir -p ${DATA_DIR}/generalization
python scripts/generalization/classifier_mnist_generalization_train.py ${EXTRA_ARGS} --save-dir ${DATA_DIR} &
python scripts/generalization/classifier_mnist_generalization_train.py ${EXTRA_ARGS} --dropout 0.5 --save-dir ${DATA_DIR} &
python scripts/generalization/classifier_mnist_generalization_train.py ${EXTRA_ARGS} --random-labels --save-dir ${DATA_DIR} &
python scripts/generalization/classifier_mnist_generalization_train.py ${EXTRA_ARGS} --random-pixels --save-dir ${DATA_DIR} &
for regularizer in l1 l2; do
  for regularize in kernel activity; do
    python scripts/generalization/classifier_mnist_generalization_train.py ${EXTRA_ARGS} --regularize $regularize --regularizer $regularizer --save-dir ${DATA_DIR} &
  done
done
wait
