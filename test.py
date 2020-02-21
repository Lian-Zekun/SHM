import segmentation_models_pytorch as smp

model = smp.PSPNet('resnet50', classes=3)
print(model)
