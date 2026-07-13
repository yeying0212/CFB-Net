# Test CFB-Net on S1GFloods
python ./train_test_tools/test.py --file_root S1G --lr 5e-4 --max_steps 26800

# Test CFB-Net on ETCI-2021
python ./train_test_tools/test.py --file_root etci --lr 5e-4 --max_steps 100000

# Test CFB-Net on UrbanSARFloods
python ./train_test_tools/test.py --file_root URBAN --lr 5e-4 --max_steps 30000


