<template>
  <div>
    <nav class="navbar navbar-light bg-light">
      <div class="d-flex justify-content-end w-100">
        <button class="btn btn-outline-primary mr-2" type="button">
          <i class="fas fa-user"></i> {{ username }}
        </button>
      </div>
    </nav>
    <div class="container mt-4">
      <div class="form-group mb-3">
        <label for="fileUpload">Dataset: </label>
        <input type="file" class="form-control-file" id="fileUpload" @change="handleFileUpload" accept=".csv">
      </div>
      <button class="btn btn-primary" @click="uploadFile" :disabled="loading">Subir</button>
      <div v-if="loading" class="spinner-border" role="status">
        <span class="sr-only">Loading...</span>
      </div>
      <div v-if="notification" class="alert alert-success" role="alert">
        {{ notification }}
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'UploadComponent',
  data() {
    return {
      username: '',
      selectedFile: null,
      loading: false,
      notification: ''
    };
  },
  methods: {
    fetchUsername() {
      axios.get(process.env.VUE_APP_BACKEND_URL + '/user')
          .then(response => {
            this.username = response.data.username;
          })
          .catch(error => {
            console.error('Error fetching username:', error);
          });
    },
    handleFileUpload(event) {
      const file = event.target.files[0];
      if (file && file.type === 'text/csv') {
        this.selectedFile = file;
      } else {
        alert('Please upload a valid CSV file.');
        this.selectedFile = null;
      }
    },
    showNotification(message) {
      this.notification = message;
      setTimeout(() => {
        this.notification = '';
      }, 3000);
    },
    uploadFile() {
      if (!this.selectedFile) {
        alert('Please select a file first.');
        return;
      }

      this.loading = true;
      const formData = new FormData();
      formData.append('file', this.selectedFile);

      axios.post(process.env.VUE_APP_BACKEND_URL + '/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
          .then(() => {
            this.showNotification('Se subió exitosamente el archivo.');
          })
          .catch(error => {
            console.error('Error uploading file:', error);
          })
          .finally(() => {
            this.loading = false;
          });
    }
  },
  created() {
    this.fetchUsername();
  }
}
</script>