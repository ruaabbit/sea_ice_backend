class Configs:
    def __init__(self):
        # self.model = "IceMamba"
        self.model = "SimVP"
        # self.model = "SwinIP"

        # Paths
        self.full_data_path = {
            "sic": "seaice/cross_modality/data/sic_path.txt",
            "siv_u": "seaice/cross_modality/data/siv_u_path.txt",
            "siv_v": "seaice/cross_modality/data/siv_v_path.txt",
            "u10": "seaice/cross_modality/data/u10_path.txt",
            "v10": "seaice/cross_modality/data/v10_path.txt",
            "t2m": "seaice/cross_modality/data/t2m_path.txt",
            "max": "seaice/cross_modality/data/max_values.npy",
            "min": "seaice/cross_modality/data/min_values.npy",
        }
        self.train_log_path = "train_logs"
        self.test_results_path = "test_results"

        # Trainer related
        self.batch_size = 2
        self.batch_size_vali = 4
        self.lr = 1e-5
        self.num_epochs = 300
        self.patience = self.num_epochs // 10
        self.num_workers = 4

        # Data related
        self.img_size = (448, 304)  # 原始的(H, W)

        self.input_dim = 6  # 输入张量对应的通道数
        self.input_length = 7  # 每轮训练输入多少张数据,T
        self.pred_length = 7  # 每轮训练输出多少张数据,T
        self.input_gap = 1  # 每张输入数据之间的间隔
        self.pred_gap = 1  # 每张输出数据之间的间隔
        self.pred_shift = self.pred_gap * self.pred_length

        self.train_period = (19790101, 20151231)
        self.eval_period = (20160101, 20201231)

        # model related
        if self.model == "IceMamba":
            self.embed_dim = 512
            self.patch_size = 1
            self.N_S = 2
            self.hid_S = 64
            self.kernels = [3, 5, 7, 9]
            self.attn_drop = 0.0
            self.pos_drop = 0.0
            self.d_state = 64
            self.d_conv = 4
            self.expand = 2
            self.headdim = 32

        elif self.model == "SimVP":
            self.hid_S = 32
            self.hid_T = 256
            self.N_S = 2
            self.N_T = 8
            self.spatio_kernel_enc = 3
            self.spatio_kernel_dec = 3

        elif self.model == "SwinIP":
            self.hidden_channels = 32
            self.embed_dim = 128
            self.patch_size = (1, 1)
            self.num_encoder_layers = 2
            self.num_layers = 8
            self.spatio_kernel_enc = 3
            self.spatio_kernel_dec = 3


# Instantiate the configuration object
configs = Configs()
