mkdir -p data/generalization
python scripts/generalization/classifier_mnist_generalization_train.py &
python scripts/generalization/classifier_mnist_generalization_train.py --dropout 0.5 &
python scripts/generalization/classifier_mnist_generalization_train.py --scrambled &
for regularizer in l1 l2; do
  for regularize in kernel activity; do
    python scripts/generalization/classifier_mnist_generalization_train.py --regularize $regularize --regularizer $regularizer &
  done
done
wait
