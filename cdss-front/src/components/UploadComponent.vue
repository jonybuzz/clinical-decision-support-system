<template>
  <div>
    <nav class="navbar navbar-light bg-light">
      <div class="d-flex justify-content-end w-100"> <!-- Flex container for right alignment -->
        <button class="btn btn-outline-primary mr-2" type="button"> <!-- Button with margin -->
          <i class="fas fa-user"></i> {{ username }}
        </button>
      </div>
    </nav>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'UploadComponent',
  data() {
    return {
      username: ''
    };
  },
  methods: {
    fetchUsername() {
      axios.get(process.env.VUE_APP_BACKEND_URL + '/user')
        .then(response => {
          this.username = response.data.username; // Set the fetched username
        })
        .catch(error => {
          console.error('Error fetching username:', error);
        });
    }
  },
  created() {
    this.fetchUsername(); // Fetch the username when the component is created
  }
}
</script>
