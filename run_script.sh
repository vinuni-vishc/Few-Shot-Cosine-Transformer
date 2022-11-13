#####################################################################
# "Conv" backbones take ~15 min to run each training time
# "Resnet" backbones take ~40 min to run each training time
# These training times vary depending on the dataset
#####################################################################

for dataset in "miniImagenet" #"CUB" "CIFAR" "Yoga"
do
    for way in 5
    do
        for backbone in  "ResNet34" #"ResNet18" Conv4" "Conv6" 
        do
            for shot in 1 5
            do
                for method in "FSCT_softmax" "FSCT_cosine"
                do
                    for aug in 0 1
                    do
                        python train_test.py --dataset $dataset --backbone $backbone --wandb 1 --n_way $way --k_shot $shot --method $method --train_aug $aug
                    done
                done
                for method in "CTX_softmax" "CTX_cosine" # Our paper did not experiment CTX methods with augmentation
                do
                    python train_test.py --dataset $dataset --backbone $backbone --wandb 1 --n_way $way --k_shot $shot --method $method
                done
            done
        done
    done
done
