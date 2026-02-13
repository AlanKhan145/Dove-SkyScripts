# 📊 SkyScript Dataset Documentation

Tài liệu chi tiết về **SkyScript** - Large and Semantically Diverse Vision-Language Dataset for Remote Sensing.

---

## 📖 Tổng quan

SkyScript là bộ dataset vision-language lớn nhất và đa dạng nhất cho remote sensing images, được xây dựng bằng cách kết nối:

- **Google Earth Engine (GEE)**: Nguồn ảnh vệ tinh/aerial
- **OpenStreetMap (OSM)**: Nguồn semantic tags

### 📈 Thống kê

| Metric | Value |
|--------|-------|
| **Total image-text pairs** | 5.2M (unfiltered) → 2.6M (filtered top 50%) |
| **Distinct semantic tags** | 44K (unfiltered) → 29K (filtered) |
| **Distinct single-object captions** | 100K |
| **Distinct multi-object captions** | 1.2M |
| **Ground Sampling Distance (GSD)** | 0.1m - 30m |
| **Geographic coverage** | Global (except Antarctica) |
| **Tag accuracy** | 96.1% (manually verified) |

---

## 🌍 Geographic Coverage

### Distribution

```
North America    ████████████████ 35%
Europe          ███████████████  32%
Asia            ███████          15%
South America   ████              8%
Africa          ███               6%
Oceania         ██                4%
Antarctica      -                 0%
```

### Coverage Map

Các khu vực có mật độ cao:
- **United States**: High-resolution imagery (0.1m - 1m)
- **Europe** (Switzerland, Spain, Germany, Finland): Very high resolution
- **Urban areas**: Denser object annotations
- **Rural areas**: Lower density but still covered

**Lưu ý**: High-resolution images (<1m GSD) chủ yếu tập trung ở U.S. và Europe do giới hạn về licensing.

---

## 🗂️ Data Sources

### Image Collections (from Google Earth Engine)

| Collection | GSD (m) | Country | Resolution |
|------------|---------|---------|------------|
| SWISSIMAGE 10cm RGB | 0.1 | Switzerland | Very High |
| Spain RGB orthophotos | 0.1 | Spain | Very High |
| Brandenburg RGBN orthophotos | 0.2 | Germany | Very High |
| Finland RGB NLS orthophotos | 0.5 | Finland | High |
| NAIP (National Agriculture Imagery Program) | 0.6-1 | U.S. | High |
| Planet SkySat Public Ortho Imagery (RGB) | 0.8 | Global | High |
| Planet SkySat Public Ortho Imagery (MS) | 2 | Global | Medium |
| Harmonized Sentinel-2 MSI, Level-2A | 10 | Global | Medium |
| Landsat 8 Collection 2 Tier 1 TOA | 30 | Global | Low |
| Landsat 9 Collection 2 Tier 1 TOA | 30 | Global | Low |

### Semantic Tags (from OpenStreetMap)

OSM tags được phân loại thành:

1. **Visually groundable tags**: Có thể nhìn thấy trong ảnh
2. **Non-visually groundable tags**: Không thể nhìn thấy (ví dụ: house number)

**Tag Classification Process**:

```
All OSM tags
    ↓
Stage 1: Visually groundable? (Binary Classification)
    ├─ Yes → Stage 2
    └─ No → Exclude
         ↓
Stage 2: Maximum GSD? (Multi-class Classification)
    ├─ 0.1m → Suitable for very high-res images
    ├─ 0.2m → Suitable for high-res images
    ├─ 0.6m
    ├─ 1m
    └─ 10m → Suitable for all images
```

---

## 📋 Dataset Structure

### File Format

Dataset được phân phối dưới dạng:

```
skyscript/
├── images/
│   ├── images2.zip → images2/ (650K images)
│   ├── images3.zip → images3/ (650K images)
│   ├── images4.zip → images4/ (650K images)
│   ├── images5.zip → images5/ (650K images)
│   ├── images6.zip → images6/ (650K images)
│   └── images7.zip → images7/ (650K images)
│
├── meta/
│   ├── meta2.zip → meta2/ (metadata pickles)
│   ├── meta3.zip → meta3/
│   ├── meta4.zip → meta4/
│   ├── meta5.zip → meta5/
│   ├── meta6.zip → meta6/
│   └── meta7.zip → meta7/
│
└── dataframe/
    ├── SkyScript_train_unfiltered_5M.csv (5.1M pairs)
    ├── SkyScript_train_top50pct_filtered_by_CLIP_openai.csv (2.6M pairs)
    ├── SkyScript_train_top30pct_filtered_by_CLIP_openai.csv (1.5M pairs)
    ├── SkyScript_val_5K_filtered_by_CLIP_openai.csv (5K pairs)
    └── SkyScript_test_30K_filtered_by_CLIP_openai.csv (30K pairs)
```

### Image Naming Convention

Mỗi image file có tên theo format: `{object_id}_{source_alias}_{year}.jpg`

**Ví dụ**: `a198234555_CH_19.jpg`
- `a`: Object type (a=area/polygon, w=way/polyline, n=node/point)
- `198234555`: OSM object ID
- `CH`: Image source alias (SWISSIMAGE 10cm)
- `19`: Capture year (2019)

### CSV Format

Mỗi CSV file chứa các cột:

| Column | Description | Example |
|--------|-------------|---------|
| `filepath` | Relative path to image | `images2/a198234555_CH_19.jpg` |
| `title` | Single-object caption | `road of residential, surface of asphalt` |
| `title_multi_objects` | Multi-object caption | `road of residential with surface of asphalt, surrounded by landuse of residential` |

### Metadata Pickle Format

Mỗi `.pickle` file chứa dictionary với:

```python
{
    'box': (west_lon, south_lat, east_lon, north_lat),  # Bounding box
    'time': (year, month, day, hour, minute),            # Acquisition time
    'center_tags': {                                      # Focus object tags
        'highway': 'residential',
        'surface': 'asphalt'
    },
    'surrounding_tags': [                                 # Surrounding objects
        {'landuse': 'residential'},
        {'natural': 'tree'},
        ...
    ]
}
```

---

## 🏷️ Semantic Tags

### Tag Categories

**Top 20 Most Frequent Tags**:

| Rank | Tag | Count | Category |
|------|-----|-------|----------|
| 1 | highway:residential | 450K | Road |
| 2 | building | 380K | Building |
| 3 | landuse:residential | 320K | Land Use |
| 4 | natural:tree | 280K | Nature |
| 5 | waterway:stream | 250K | Water |
| 6 | landuse:farmland | 220K | Land Use |
| 7 | highway:service | 210K | Road |
| 8 | amenity:parking | 190K | Amenity |
| 9 | natural:water | 180K | Nature |
| 10 | highway:footway | 170K | Road |
| 11 | power:line | 160K | Infrastructure |
| 12 | railway | 150K | Transportation |
| 13 | landuse:forest | 140K | Land Use |
| 14 | building:house | 130K | Building |
| 15 | highway:track | 120K | Road |
| 16 | amenity:school | 110K | Amenity |
| 17 | leisure:park | 100K | Leisure |
| 18 | man_made:bridge | 95K | Man-made |
| 19 | landuse:industrial | 90K | Land Use |
| 20 | natural:coastline | 85K | Nature |

### Tag Diversity

```
Tags with ≥1,000 images:  580 tags
Tags with ≥100 images:    1,800 tags
Tags with ≥10 images:     6,000+ tags
Total unique tags:        29,000 tags
```

### Example Tags by Category

**Roads (highway)**:
- `motorway`, `trunk`, `primary`, `secondary`, `residential`, `service`, `track`, `footway`, `cycleway`
- Attributes: `surface` (asphalt, concrete, gravel), `lanes`, `width`, `oneway`

**Buildings**:
- Types: `house`, `apartments`, `commercial`, `industrial`, `church`, `school`, `hospital`
- Attributes: `roof:shape` (flat, gabled, hipped), `building:levels`, `building:material`

**Land Use**:
- `residential`, `commercial`, `industrial`, `farmland`, `forest`, `meadow`, `orchard`, `vineyard`
- Attributes: `crop` type for farmland

**Natural Features**:
- `water`, `tree`, `wood`, `grassland`, `scrub`, `wetland`, `beach`, `glacier`

**Infrastructure**:
- Power: `line`, `pole`, `tower`, `plant`, `substation`, `generator`
- Water: `reservoir`, `storage_tank`, `wastewater_plant`
- Transport: `railway`, `airport`, `helipad`, `harbour`

---

## 📸 Caption Examples

### Single-Object Captions

```
Example 1:
Image: Aerial view of a solar farm
Caption: "power plant, plant source of solar, plant method of photovoltaic"

Example 2:
Image: Farmland with crops
Caption: "landuse of farmland, crop of wheat"

Example 3:
Image: Residential road
Caption: "road of residential, surface of asphalt, lanes of 2"

Example 4:
Image: Storage tanks
Caption: "man made storage tank"
```

### Multi-Object Captions

```
Example 1:
Image: Solar farm surrounded by roads
Caption: "power plant with plant source of solar and plant method of photovoltaic, 
         surrounded by road of service; landuse of industrial"

Example 2:
Image: School with parking lot
Caption: "amenity of school, surrounded by amenity of parking; road of service"

Example 3:
Image: Bridge over river
Caption: "man made bridge, surrounded by waterway of river"

Example 4:
Image: Quarry in forest area
Caption: "landuse of quarry with resource of limestone, 
         surrounded by landuse of forest"
```

---

## 🔍 Data Quality

### Filtering Process

**Step 1: Tag Classification**
- Binary classification: Visually groundable? (F1 = 0.88)
- Multi-class classification: Maximum GSD? (Accuracy = 0.53)

**Step 2: Image-Text Similarity Filtering**
- Compute CLIP similarity for all pairs
- Keep top X% (20%, 30%, or 50%)
- Remove low-correlation pairs

**Common Noise Sources**:
1. ❌ Image not fully loaded
2. ❌ Object obscured by trees/clouds
3. ❌ Object built after image capture
4. ❌ Incorrect OSM annotation
5. ❌ Temporal mismatch

### Quality Metrics

| Subset | Pairs | Avg. CLIP Similarity | Manual Validation |
|--------|-------|---------------------|-------------------|
| **Top 20%** | 1.0M | 0.31 | 98.5% accurate |
| **Top 30%** | 1.5M | 0.29 | 97.2% accurate |
| **Top 50%** | 2.6M | 0.26 | 96.1% accurate |
| Full (unfiltered) | 5.2M | 0.22 | ~85% accurate |

**Recommendation**: Sử dụng **top 50%** cho best trade-off giữa quality và quantity.

---

## 📥 Download Instructions

### Option 1: Using download script

```bash
# Download images
bash download_skyscript.sh

# Unzip
bash unzip_skyscript.sh
```

### Option 2: Manual download

```bash
# Download images (6 parts)
for i in {2..7}; do
    curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/images${i}.zip
    unzip images${i}.zip -d data/images/
done

# Download metadata (6 parts)
for i in {2..7}; do
    curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/meta${i}.zip
    unzip meta${i}.zip -d data/meta/
done

# Download CSV files
curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/dataframe/SkyScript_train_top50pct_filtered_by_CLIP_openai.csv
curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/dataframe/SkyScript_val_5K_filtered_by_CLIP_openai.csv
curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/dataframe/SkyScript_test_30K_filtered_by_CLIP_openai.csv
```

### Storage Requirements

| Component | Size | Description |
|-----------|------|-------------|
| Images (6 parts) | ~150 GB | Compressed JPG images |
| Metadata (6 parts) | ~5 GB | Pickle files |
| CSV files | ~500 MB | Image-text pairs |
| **Total** | **~155 GB** | Full dataset |

---

## 🔧 Data Loading

### PyTorch Dataset

```python
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import pickle

class SkyScriptDataset(Dataset):
    def __init__(self, csv_path, data_root, caption_field='title', transform=None):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.caption_field = caption_field
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.data_root, row['filepath'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get caption
        caption = row[self.caption_field]
        
        # Optional: Load metadata
        meta_path = img_path.replace('.jpg', '.pickle').replace('images', 'meta')
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return {
            'image': image,
            'caption': caption,
            'metadata': metadata
        }
```

### Usage

```python
from torchvision import transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = SkyScriptDataset(
    csv_path='data/dataframe/SkyScript_train_top50pct_filtered_by_CLIP_openai.csv',
    data_root='data/images',
    caption_field='title_multi_objects',
    transform=transform
)

# Create dataloader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

---

## 📊 Dataset Statistics

### Image Resolution Distribution

```
0.1m GSD:   ███████           (15%)  - Very detailed
0.2m GSD:   ██████            (12%)
0.5m GSD:   █████             (10%)
0.6-1m GSD: ████████          (18%)  - High detail
2m GSD:     ████              (8%)
10m GSD:    ███████████       (22%)  - Medium detail
30m GSD:    ███████           (15%)  - Low detail
```

### Caption Length Distribution

```
Single-object captions:
Mean: 8.5 words
Median: 7 words
Range: 2-25 words

Multi-object captions:
Mean: 15.3 words
Median: 13 words
Range: 5-50 words
```

### Object Categories Distribution

```
Transportation  ████████████████ (28%) - Roads, railways, airports
Buildings       ██████████████   (24%) - Houses, schools, commercial
Land Use        ███████████      (19%) - Farmland, forest, residential
Natural         ████████         (14%) - Water, trees, coastline
Infrastructure  ██████           (10%) - Power lines, bridges, dams
Other           ███              (5%)  - Amenities, leisure
```

---

## 🎓 Citation

```bibtex
@article{wang2023skyscript,
  title={SkyScript: A Large and Semantically Diverse Vision-Language Dataset for Remote Sensing},
  author={Wang, Zhecheng and Prabha, Rajanie and Huang, Tianyuan and Wu, Jiajun and Rajagopal, Ram},
  journal={arXiv preprint arXiv:2312.12856},
  year={2023}
}
```

---

## 📄 License

Dataset được phân phối dưới **MIT License**. Images từ Google Earth Engine tuân theo licensing terms của từng image collection.

---

## 🔗 Related Resources

- [Official SkyScript Repository](https://github.com/wangzhecheng/SkyScript)
- [Google Earth Engine](https://earthengine.google.com/)
- [OpenStreetMap](https://www.openstreetmap.org/)
- [Paper on arXiv](https://arxiv.org/abs/2312.12856)
