# without pretrain
python run_epoch.py --pre_plant ar --model revolution

# with pretrain
python run_pre.py --pre_plant ar  # pre-train
python run_epoch.py --pre_plant ar --model revolution --pre_training --pre_froze  # probing
python run_epoch.py --pre_plant ar --model revolution --pre_training              # fine-tuning
python run_epoch.py --pre_plant ar --model revolution --pre_two                   # combination
