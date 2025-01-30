from data_utils import DataUtils, Data_Utils_Config

shard_util = DataUtils (Data_Utils_Config)

shard_util.process_images_in_folder("./clones", "./contextRenderShards", 800)