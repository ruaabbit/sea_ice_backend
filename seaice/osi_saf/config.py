class Configs:
    def __init__(self):
        self.model = "SICFN"

        # trainer related
        self.batch_size_vali = 16
        self.batch_size = 4
        self.lr = 1e-3
        self.weight_decay = 1e-2
        self.num_epochs = 200
        self.early_stop = True
        self.patience = self.num_epochs // 10
        self.gradient_clip = True
        self.clip_threshold = 1.0
        self.layer_norm = False
        self.num_workers = 4

        # data related
        self.img_size = (432, 432)

        self.input_dim = 1  # input_dim: 输入张量对应的通道数，对于彩图为3，灰图为1。
        self.output_dim = 1  # output_dim: 输入张量对应的通道数，对于彩图为3，灰图为1。

        self.input_length = 14  # 每轮训练输入多少张数据
        self.pred_length = 14  # 每轮训练输出多少张数据

        self.input_gap = 1  # 每张输入数据之间的间隔
        self.pred_gap = 1  # 每张输出数据之间的间隔

        self.pred_shift = self.pred_gap * self.pred_length

        self.train_period = (19910101, 20100101)
        self.eval_period = (20100101, 20151231)

        # model related
        self.kernel_size = (3, 3)
        self.patch_size = (2, 2)
        self.hidden_dim = (
            96,
            96,
            96,
            96,
        )  # hidden_dim: 隐藏状态的神经单元个数，也就是隐藏层的节点数，应该可以按计算需要设置。

        self.decouple_beta = 0.1  # PredRNNv2

        self.kernel_size_3D = (2, 2, 2)  # E3DLSTM

        # SimVP
        self.hid_S = 64
        self.hid_T = 256
        self.N_T = 8
        self.N_S = 4
        self.spatio_kernel_enc = 3
        self.spatio_kernel_dec = 3

        # TAU
        self.mlp_ratio = 4.0
        self.drop = 0.05
        self.drop_path = 0.05

        # ConvNeXt
        self.use_grn = True

        # SICFN
        self.patch_embed_size = (8, 8)
        self.dropcls = 0.05
        self.fno_blocks = 8
        self.fno_bias = True
        self.fno_softshrink = 0.05

        # paths
        self.data_paths = "data/data_path.txt"
        self.train_log_path = "train_logs"
        self.test_results_path = "test_results"


configs = Configs()
