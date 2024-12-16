import json

def get_num_classes(meta_file):
    with open(meta_file, 'r') as f:
        data = json.load(f)
    return len(data['classes'])

meta_file = '/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/datasetkaggle/Teeth Segmentation JSON/meta.json'
meta_file2="/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/datasetkaggle/Teeth Segmentation PNG/meta.json"
num_classes = get_num_classes(meta_file)
num_classes2=get_num_classes(meta_file2)
print(f"Toplam Sınıf Sayısı(Teeth Segmentation JSON): {num_classes} ")
print(f"Toplam Sınıf Sayısı(Teeth Segmentation PNG): {num_classes2} ")