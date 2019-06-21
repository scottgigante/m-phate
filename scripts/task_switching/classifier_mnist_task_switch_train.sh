mkdir -p data/task_switch
for scheme in task domain class; do
    for optimizer in adam adagrad; do
        python scripts/task_switching/classifier_mnist_task_switch_train.py --optimizer $optimizer $scheme &
        python scripts/task_switching/classifier_mnist_task_switch_train.py --optimizer $optimizer --rehearsal 5000 $scheme &
    done
done
wait
