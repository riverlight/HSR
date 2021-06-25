train.h5 : D:\\workroom\\tools\\dataset\\SR\\qnSR_DS\\train
eval.h5 : D:\\workroom\\tools\\dataset\\SR\\qnSR_DS\\eval

T91.h5 : D:\\workroom\\tools\\dataset\\SR\\T91-image\\T91\\T91
Set5.h5 : D:\workroom\tools\dataset\SR\srgan\Set5

hr.h5: 只包含 hr 数据，使用 ntire20 track1 的 clean-up train 数据制作

vsr-train.h5: 跟 hr.h5 类似，增加了一些人脸的image
vsr-val.h5: 由 ntire20 track1 的 val-gt 数据构成(100 images)，同时增加了10张人脸的image

vsr_train_hwcbgr.h5: 跟 vsr-train.h5 的数据源是完全一样的，但是格式改成了 hwcbgr，这样似乎更方便处理，在训练的时候可以提升性能
vsr_val_hwcbgr.h5: 跟 vsr-val.h5 的数据源是完全一样的，格式改成了 hwcbgr
http://test-v.oss-cn-shanghai.aliyuncs.com/private/vsr_train_hwcbgr.h5
http://test-v.oss-cn-shanghai.aliyuncs.com/private/vsr_val_hwcbgr.h5

vsr_dy_train_hwcbgr.h5 和 vsr_dy_val_hwcbgr.h5：用的是抖音的短视频，格式是 hwcbgr
http://test-v.oss-cn-shanghai.aliyuncs.com/private/vsr_dy_train_hwcbgr.h5
http://test-v.oss-cn-shanghai.aliyuncs.com/private/vsr_dy_val_hwcbgr.h5
