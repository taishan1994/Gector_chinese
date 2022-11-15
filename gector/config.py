class Args:
    bert_dir = "../model_hub/chinese-bert-wwm-ext"
    output_dir = "./checkpoints/"
    ctc_vocab_dir = "./data/ctc_vocab/"
    detect_tags_file = "ctc_detect_tags.txt"
    log_dir = "./logs/"

    data_name = "cail2022"
    model_name = "roberta"
    correct_tags_file = "ctc_correct_sighan13_tags.txt"

    warmup_proportion = 0.01
    learning_rate = 3e-5
    adam_epsilon = 1e-8
    seed = 999
    max_seq_len = 128
    use_tensorboard = False
    batch_size = 64
    epochs = 30
    eval_step = 100
    max_grad_norm = 1.0

    detect_vocab_size = 2
    correct_vocab_size = 20675

    # midu2022
    midu_train_path = "./data/midu2022/preliminary_extend_train.json"
    midu_dev_path = "./data/midu2022/preliminary_val.json"
    midu_detect_tags_path = "ctc_detect_tags.txt"
    midu_correct_tags_path = "ctc_correct_tags.txt"

    # sighan13
    sighan13_train_path = "./data/sighan13/train.json"
    sighan13_dev_path = "./data/sighan13/test.json"
    sighan13_detect_tags_path = "ctc_detect_tags.txt"
    sighan13_correct_tags_path = "ctc_correct_sighan13_tags.txt"

    # sighan14
    sighan14_train_path = "./data/sighan14/train.json"
    sighan14_dev_path = "./data/sighan14/test.json"
    sighan14_detect_tags_path = "ctc_detect_tags.txt"
    sighan14_correct_tags_path = "ctc_correct_sighan14_tags.txt"

    # sighan15
    sighan15_train_path = "./data/sighan15/train.json"
    sighan15_dev_path = "./data/sighan15/test.json"
    sighan15_detect_tags_path = "ctc_detect_tags.txt"
    sighan15_correct_tags_path = "ctc_correct_sighan15_tags.txt"

    # sighan15_2
    sighan15_2_train_path = "./data/sighan15_2/train.json"
    sighan15_2_dev_path = "./data/sighan15_2/test.json"
    sighan15_2_detect_tags_path = "ctc_detect_tags.txt"
    sighan15_2_correct_tags_path = "ctc_correct_sighan15_2_tags.txt"

    # cail2022
    cail2022_train_path = "./data/cail2022/train.json"
    cail2022_dev_path = "./data/cail2022/test.json"
    cail2022_detect_tags_path = "ctc_detect_tags.txt"
    cail2022_correct_tags_path = "ctc_correct_cail2022_tags.txt"