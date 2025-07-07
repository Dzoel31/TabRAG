# TabRAG 🚀

TabRAG (Tanya Bareng RAG) adalah aplikasi Streamlit yang memungkinkan pengguna mengunggah dokumen dan menanyakannya menggunakan pendekatan RAG (Retrieval-Augmented Generation) dengan backend Elasticsearch.

## Fitur ✨

- **Unggah Dokumen** 📄: Pengguna dapat mengunggah dokumen dalam format PDF.
- **Query Dokumen** ❓: Pengguna dapat mengajukan pertanyaan terkait dokumen yang telah diunggah.
- **Backend Elasticsearch** 🔎: Menggunakan Elasticsearch untuk pencarian dan pengambilan informasi dari dokumen.
- **Antarmuka Streamlit** 🖥️: Menyediakan UI yang sederhana dan intuitif untuk interaksi pengguna.

## Instalasi 🛠️

Sebelum melanjutkan proses instalasi, pastikan Anda telah menginstal beberapa aplikasi berikut:

- **Python 3.12** atau lebih baru 🐍
- **Elasticsearch**:
  - Instal Elasticsearch sesuai dengan [panduan resmi](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html).
  - Pastikan Elasticsearch sudah berjalan pada `http://localhost:9200`.
- **Ollama** 🤖:
  - Pastikan Anda telah menginstal dan mengonfigurasi Ollama untuk model yang digunakan.

1. Clone repositori ini:

        ```bash
        git clone https://github.com/dzoel31/TabRAG.git
        cd TabRAG
        ```

2. Instal dependensi (memerlukan uv, bisa unduh dari [link ini](https://docs.astral.sh/uv/getting-started/installation/)):

        ```bash
        uv sync
        ```

## Penggunaan ▶️

1. Jalankan aplikasi Streamlit:

        ```bash
        streamlit run main.py
        ```

2. Buka browser dan akses `http://localhost:8501` untuk mulai menggunakan aplikasi.

## Kontribusi 🤝

Jika Anda ingin berkontribusi pada proyek ini, silakan ikuti langkah-langkah berikut:

1. Fork repositori ini.
2. Buat cabang fitur baru (`git checkout -b fitur-baru`).
3. Lakukan perubahan yang diinginkan.
4. Uji perubahan Anda.
5. Kirim permintaan tarik (pull request) ke repositori utama.

## Commit Convention

Follow this format for commit messages:

- **feat**: Add a new feature
- **fix**: Fix a bug
- **docs**: Documentation changes
- **ref**: Code refactoring that does not add features or fix bugs
- **test**: Add or improve tests
- **chore**: Minor changes (build, dependencies, etc.)

Examples:

- feat: add parser for DOCX files
- fix: fix error when uploading large PDF
- docs: update README for usage instructions

## Dalam Pengembangan 🏗️

- Testing.
- Dokumentasi.
- Mendukung lebih banyak format dokumen.
- Mendukung opsi konfigurasi tambahan untuk model LLM.
- Peningkatan UI/UX.
- Support untuk berbagai sistem operasi.

## Issue 🐞

Jika Anda menemukan bug atau memiliki saran untuk peningkatan, silakan buka [issue](https://github.com/dzoel31/TabRAG/issues).

---

**Status Proyek:** Ready for Use 🚦  
**Terakhir Diperbarui:** 7 Juli 2025 📅  
**Versi:** 0.1.0 🏷️
