# Eksperimen: Klasifikasi Senyawa Aktif HIV

## Panduan Menjalankan Eksperimen

Panduan eksperimen ini dilakukan dengan menggunakan python=3.12 dan CUDA=12.2. Untuk versi lain silahkan dapat disesuaikan.

Clone repositori ini

```bash
git clone https://github.com/izzudd/interna-hiv.git
cd interna-hiv
```

Buat environment python baru dengan `conda`

```bash
conda create -n hiv python=3.12 -y
conda activate hiv
```

Install package yang diperlukan

```bash
pip install tensorflow[and-cuda]==2.16.1
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip install ipykernel pandas matplotlib numpy deepchem rdkit dgllife pyyaml pydantic
```

Setting environment variable unuk DGL

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'NVIDIA_DIR=$(dirname $(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")))
for dir in $NVIDIA_DIR/*; do
  if [ -d "$dir/lib" ]; then
    export LD_LIBRARY_PATH="$dir/lib:$LD_LIBRARY_PATH"
  fi
done' > $CONDA_PREFIX/etc/conda/activate.d/activate.sh
echo 'export GSETTINGS_SCHEMA_DIR_CONDA_BACKUP="${GSETTINGS_SCHEMA_DIR:-}"
export GSETTINGS_SCHEMA_DIR="$CONDA_PREFIX/share/glib-2.0/schemas"' > $CONDA_PREFIX/etc/conda/activate.d/libglib_activate.sh
```

Daftar model dan notebook eksperimen dapat dilihat pada bagian [eksperimentasi](#eksperimentasi). Sebelum menjalankan notebook pastikan untuk menggunakan environment `hiv`.

## Tentang Dataset

Sumber data utama: [MoleculeNet HIV Dataset](https://raw.githubusercontent.com/deepchem/deepchem/master/examples/hiv/HIV.csv)
Referensi pendukung: [National Cancer Institute](https://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data)

Data utama yang akan digunakan pada percobaan ini diambil dari laman MoleculeNet. Menurut referensi data pendukung, dataset ini berisi kumpulan senyawa beserta reaksinya terhadap virus HIV. Tujuan dan potensi penggunaan dataset ini adalah sebagai berikut:

- Mengidentifikasi senyawa baru dengan properti anti-HIV: Para peneliti dapat menggunakan data ini untuk mencari senyawa yang memiliki potensi untuk menghambat replikasi HIV. Senyawa ini kemudian dapat dikembangkan menjadi obat-obatan baru untuk melawan HIV/AIDS.
- Mempelajari mekanisme obat anti-HIV: Para peneliti dapat menggunakan data ini untuk mempelajari bagaimana obat anti-HIV bekerja. Informasi ini dapat membantu para peneliti dalam mengembangkan obat-obatan baru yang lebih efektif dan memiliki efek samping yang lebih sedikit.
- Mengembangkan model prediktif untuk aktivitas anti-HIV: Para peneliti dapat menggunakan data ini untuk mengembangkan model prediktif yang dapat digunakan untuk memprediksi aktivitas anti-HIV dari senyawa baru. Model ini dapat membantu para peneliti dalam mempercepat proses pengembangan obat-obatan baru.

## Fitur Dataset

Data pada sumber data utama sudah diproses sedemikian rupa sehingga menghasilkan 3 fitur pada dataset, diantaranya:

1. smiles (string): Senyawa yang digunakan dalam notasi standar SMILES
2. activity (ordinal): Aktivitas senyawa bereaksi dengan virus HIV terbagi menjadi 3 kategori yaitu
   1. CA (Confirmed Active - Aktif Terkonfirmasi): Senyawa dikonfirmasi memiliki aktivitas antivirus yang kuat terhadap HIV menurut hasil percobaan.
   2. CM (Confirmed Moderately Active - Aktif Moderat Terkonfirmasi): Senyawa menunjukkan beberapa aktivitas antivirus terhadap HIV, tetapi efeknya lebih lemah dibandingkan dengan senyawa aktif terkonfirmasi.
   3. CI (Confirmed Inactive - Tidak Aktif Terkonfirmasi): Senyawa tidak menunjukkan aktivitas antivirus yang signifikan terhadap HIV dalam proses skrining. Senyawa ini tidak dianggap sebagai kandidat yang menjanjikan untuk pengembangan lebih lanjut sebagai pengobatan HIV.
3. HIV_active (biner): Pengelompokan senyawa menjadi aktif (CA dan CM) dan tidak aktif (CI) untuk menyederhanakan tugas klasifikasi.

## Eksperimentasi

Eksperimentasi model dilakukan dengan menyimpan checkpoint bobot model yang memiliki metrik f1, recall, dan roc-auc tertinggi pada data validasi. Checkpoint ini kemudian digunakan untuk melakukan testing. Model yang digunakan untuk eksperimentasi diantaranya:

1. [GCN (Graph Convolutional Network)](https://arxiv.org/abs/1609.02907): GraphModel.ipynb | GraphModel-undersampled.ipynb
2. [Attentive FP](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959): GraphModel.ipynb | GraphModel-undersampled.ipynb
3. [GAT (Graph Attention Network)](https://arxiv.org/abs/1710.10903): GraphModel.ipynb | GraphModel-undersampled.ipynb
4. [elemBERT](https://arxiv.org/abs/2309.09355): ElemBERT.ipynb | ElemBERT-undersampled.ipynb
