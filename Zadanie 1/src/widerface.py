import os
import json
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToPILImage
from PIL import Image
import zipfile
import torch
from os.path import abspath, expanduser
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import zipfile
import gdown
import requests

# Define paths
SRC_PATH: str = os.path.dirname(os.path.abspath(__file__))
DATA_PATH: str = os.path.join(os.path.dirname(SRC_PATH), "csv")
OUTPUT_PATH = os.path.join(DATA_PATH, "processed_wider_faces")
os.makedirs(OUTPUT_PATH, exist_ok=True)

def download_file_from_google_drive(file_id, root, filename, md5=None):
    """
    Download a file from Google Drive using its file ID.
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output_path = os.path.join(root, filename)
    
    if not os.path.exists(output_path):
        print(f"Downloading {filename} from Google Drive...")
        gdown.download(url, output_path, quiet=False)
    
    if md5:
        # Check if the md5 hash matches
        if not verify_md5(output_path, md5):
            raise ValueError(f"MD5 mismatch for {filename}.")
    
    return output_path

def verify_md5(filepath, expected_md5):
    """
    Verify the MD5 hash of a file.
    """
    import hashlib
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    
    file_md5 = md5_hash.hexdigest()
    return file_md5 == expected_md5

def download_and_extract_archive(url, download_root, md5=None):
    """
    Download and extract a zip archive.
    """
    zip_path = os.path.join(download_root, os.path.basename(url))
    
    if not os.path.exists(zip_path):
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    if md5:
        # Verify MD5 checksum
        if not verify_md5(zip_path, md5):
            raise ValueError(f"MD5 mismatch for {zip_path}.")
    
    # Extract the archive
    if zipfile.is_zipfile(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(download_root)
    else:
        raise ValueError(f"{zip_path} is not a valid zip file.")

class WIDERFace(VisionDataset):
    """`WIDERFace <http://shuoyang1213.me/WIDERFACE/>`_ Dataset."""

    BASE_FOLDER = "widerface"
    FILE_LIST = [
        ("15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M", "3fedf70df600953d25982bcd13d91ba2", "WIDER_train.zip"),
        ("1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q", "dfa7d7e790efa35df3788964cf0bbaea", "WIDER_val.zip"),
        ("1HIfDbVEWKmsYKJZm4lchTBDLW5N7dY5T", "e5d8f4248ed24c334bbd12f49c29dd40", "WIDER_test.zip"),
    ]
    ANNOTATIONS_FILE = (
        "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip",
        "0e3767bcf0e326556d407bf5bff5d27c",
        "wider_face_split.zip",
    )

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root=os.path.join(root, self.BASE_FOLDER), transform=transform, target_transform=target_transform)
        self.split = split
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. Please use download=True to download it.")

        self.img_info: List[Dict[str, Union[str, Dict[str, torch.Tensor]]]] = []
        if self.split in ("train", "val"):
            self.parse_train_val_annotations_file()
        else:
            self.parse_test_annotations_file()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Returns a tuple (image, target) where target is a dictionary of annotations."""
        img = Image.open(self.img_info[index]["img_path"])

        if self.transform:
            img = self.transform(img)

        target = None if self.split == "test" else self.img_info[index]["annotations"]
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.img_info)

    def parse_train_val_annotations_file(self) -> None:
        filename = "wider_face_train_bbx_gt.txt" if self.split == "train" else "wider_face_val_bbx_gt.txt"
        filepath = os.path.join(self.root, "wider_face_split", filename)

        with open(filepath) as f:
            lines = f.readlines()
            file_name_line, num_boxes_line, box_annotation_line = True, False, False
            num_boxes, box_counter = 0, 0
            labels = []
            for line in lines:
                line = line.rstrip()
                if file_name_line:
                    img_path = os.path.join(self.root, "WIDER_" + self.split, "images", line)
                    img_path = abspath(expanduser(img_path))
                    file_name_line = False
                    num_boxes_line = True
                elif num_boxes_line:
                    num_boxes = int(line)
                    num_boxes_line = False
                    box_annotation_line = True
                elif box_annotation_line:
                    box_counter += 1
                    line_split = line.split(" ")
                    line_values = [int(x) for x in line_split]
                    labels.append(line_values)
                    if box_counter >= num_boxes:
                        box_annotation_line = False
                        file_name_line = True
                        labels_tensor = torch.tensor(labels)
                        self.img_info.append(
                            {
                                "img_path": img_path,
                                "annotations": {
                                    "bbox": labels_tensor[:, 0:4].clone(),
                                    "blur": labels_tensor[:, 4].clone(),
                                    "expression": labels_tensor[:, 5].clone(),
                                    "illumination": labels_tensor[:, 6].clone(),
                                    "occlusion": labels_tensor[:, 7].clone(),
                                    "pose": labels_tensor[:, 8].clone(),
                                    "invalid": labels_tensor[:, 9].clone(),
                                },
                            }
                        )
                        box_counter = 0
                        labels.clear()
                else:
                    raise RuntimeError(f"Error parsing annotation file {filepath}")

    def parse_test_annotations_file(self) -> None:
        filepath = os.path.join(self.root, "wider_face_split", "wider_face_test_filelist.txt")
        filepath = abspath(expanduser(filepath))
        with open(filepath) as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                img_path = os.path.join(self.root, "WIDER_test", "images", line)
                img_path = abspath(expanduser(img_path))
                self.img_info.append({"img_path": img_path})

    def _check_integrity(self) -> bool:
        all_files = self.FILE_LIST.copy()
        all_files.append(self.ANNOTATIONS_FILE)
        for (_, md5, filename) in all_files:
            file, ext = os.path.splitext(filename)
            extracted_dir = os.path.join(self.root, file)
            if not os.path.exists(extracted_dir):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for (file_id, md5, filename) in self.FILE_LIST:
            download_file_from_google_drive(file_id, self.root, filename, md5)
            filepath = os.path.join(self.root, filename)
            self.custom_extract(filepath, os.path.join(self.root, filename.replace('.zip', '')))

        download_and_extract_archive(
            url=self.ANNOTATIONS_FILE[0], download_root=self.root, md5=self.ANNOTATIONS_FILE[1]
        )

    def custom_extract(self, zip_path, extract_to):
        """Extract files with long path handling."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                try:
                    target_path = os.path.join(extract_to, member)
                    if len(target_path) >= 260:
                        print(f"Skipping file with long path: {target_path}")
                        continue
                    zip_ref.extract(member, extract_to)
                except Exception as e:
                    print(f"Error extracting {member}: {e}")


# Filter images based on the number of faces
def filter_images_from_folder(dataset_root, annotations_file, min_faces=3, max_faces=5, split="train"):
    """Filters images based on the number of faces by reading from extracted folders."""
    
    # Path to the images and annotations folder
    images_folder = os.path.join(dataset_root, f"WIDER_{split}\\WIDER_{split}\\images")
    annotations_folder = os.path.join(dataset_root, "wider_face_split")
    
    # Load the annotation file for the given split (train, val, or test)
    annotation_file_path = os.path.join(annotations_folder, annotations_file)
    
    selected_images = []

    # Open the annotation file and parse it
    with open(annotation_file_path, "r") as f:
        lines = f.readlines()
        
        # Iterating over the lines in the annotation file
        file_name_line, num_boxes_line, box_annotation_line = True, False, False
        num_boxes, box_counter = 0, 0
        labels = []

        for line in lines:
            line = line.rstrip()
            if file_name_line:
                img_path = os.path.join(images_folder, line)
                file_name_line = False
                num_boxes_line = True
            elif num_boxes_line:
                num_boxes = int(line)
                num_boxes_line = False
                box_annotation_line = True
            elif box_annotation_line:
                box_counter += 1
                line_split = line.split(" ")
                line_values = [int(x) for x in line_split]
                labels.append(line_values)
                if box_counter >= num_boxes:
                    # When all bounding boxes for an image are processed
                    box_annotation_line = False
                    file_name_line = True
                    
                    # If the image contains faces within the range, process it
                    if min_faces <= len(labels) <= max_faces:
                        try:
                            img = Image.open(img_path)
                            img_info = {
                                "img_path": img_path,
                                "annotations": {
                                    "bbox": torch.tensor([label[:4] for label in labels]),
                                    "blur": torch.tensor([label[4] for label in labels]),
                                    "expression": torch.tensor([label[5] for label in labels]),
                                    "illumination": torch.tensor([label[6] for label in labels]),
                                    "occlusion": torch.tensor([label[7] for label in labels]),
                                    "pose": torch.tensor([label[8] for label in labels]),
                                    "invalid": torch.tensor([label[9] for label in labels]),
                                }
                            }
                            selected_images.append((img_info, img))
                        except Exception as e:
                            print(f"Image had too long path so it was skipped. Moving on ..")
                    # Reset for the next image
                    labels.clear()
                    box_counter = 0

    # Limit the number of images if necessary
    if len(selected_images) > 100:
        selected_images = selected_images[:100]

    return selected_images

# Save cropped faces from bounding boxes
def save_faces_from_bounding_boxes(images, output_path):
    """Saves faces cropped from bounding boxes as separate images."""
    annotations = []
    for img_info, image in images:
        # Ensure the image is in PIL format
        if isinstance(image, torch.Tensor):
            image = ToPILImage()(image)

        bboxes = img_info["annotations"]["bbox"]  # Bounding boxes from annotations
        for i, bbox in enumerate(bboxes):
            # Convert bounding box coordinates to integers
            x, y, w, h = [int(coord.item()) for coord in bbox]  # Use .item() to convert tensors to integers
            
            # Crop the face
            face = image.crop((x, y, x + w, y + h))
            
            # Construct face image filename
            face_path = os.path.join(output_path, f"image_{os.path.basename(img_info['img_path'])}_face_{i}.jpg")
            
            # Ensure the output path exists
            os.makedirs(output_path, exist_ok=True)
            
            # Save the cropped face
            face.save(face_path)
            
            # Add annotation entry for this cropped face
            annotations.append({
                "image_path": face_path,
                "attributes": {"smiling": None, "wearing_glasses": None}  # Update attributes as needed
            })

    # Save the annotations as a JSON file
    with open(os.path.join(output_path, "annotations.json"), "w") as f:
        json.dump(annotations, f, indent=4)

    print(f"Saved faces and annotations to {output_path}")


# Main execution
if __name__ == "__main__":
    print("Downloading and preparing the WIDERFace dataset...")
    dataset = WIDERFace(DATA_PATH, split="train", download=True)
    
    print("Filtering images with the appropriate number of faces...")
    annotations_file = "wider_face_train_bbx_gt.txt"  # For training split

    filtered_images = filter_images_from_folder(DATA_PATH+"\\widerface", annotations_file, min_faces=3, max_faces=5)

    
    print("Saving cropped faces and creating annotations...")
    save_faces_from_bounding_boxes(filtered_images, OUTPUT_PATH)
    
    print(f"Data preparation complete. All data saved in folder: {OUTPUT_PATH}")
