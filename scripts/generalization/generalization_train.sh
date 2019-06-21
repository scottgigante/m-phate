if [ $# -lt 1 ]; then
    DATA_DIR="./data"
else
    DATA_DIR=$1
fi

mkdir -p ${DATA_DIR}/generalization
python scripts/generalization/classifier_mnist_generalization_train.py --save-dir ${DATA_DIR} &
python scripts/generalization/classifier_mnist_generalization_train.py --dropout 0.5 --save-dir ${DATA_DIR} &
python scripts/generalization/classifier_mnist_generalization_train.py --scrambled --save-dir ${DATA_DIR} &
for regularizer in l1 l2; do
  for regularize in kernel activity; do
    python scripts/generalization/classifier_mnist_generalization_train.py --regularize $regularize --regularizer $regularizer --save-dir ${DATA_DIR} &
  done
done
wait
