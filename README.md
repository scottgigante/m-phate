# M-PHATE

Scripts for "Visualizing the PHATE of Neural Networks".

To run:

```
cd m-phate
pip install --user .
chmod +x scripts/generalization/generalization_train.sh
./scripts/generalization/generalization_train.sh
chmod +x scripts/task_switching/classifier_mnist_task_switch_train.sh
./scripts/task_switching/classifier_mnist_task_switch_train.sh
python scripts/demonstration_plot.py
python scripts/comparison_plot.py
python scripts/generalization_plot.py
python scripts/task_switch_plot.py
```