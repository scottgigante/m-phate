# M-PHATE

Scripts for "Visualizing the PHATE of Neural Networks".

To run:

```
cd m-phate
pip install --user .
DATA_DIR=./data # change this if you want to store the data elsewhere

chmod +x scripts/generalization/generalization_train.sh
chmod +x scripts/task_switching/classifier_mnist_task_switch_train.sh

./scripts/generalization/generalization_train.sh $DATA_DIR
./scripts/task_switching/classifier_mnist_task_switch_train.sh $DATA_DIR

python scripts/demonstration_plot.py $DATA_DIR
python scripts/comparison_plot.py $DATA_DIR
python scripts/generalization_plot.py $DATA_DIR
python scripts/task_switch_plot.py $DATA_DIR
```