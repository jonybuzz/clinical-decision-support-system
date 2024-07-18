<template>
  <div class="login-container">
    <form @submit.prevent="login">
      <div class="form-group mb-3">
        <label for="username">Username</label>
        <input type="text" class="form-control" id="username" v-model="credentials.username" required>
      </div>
      <div class="form-group mb-3">
        <label for="password">Password</label>
        <!-- Added @keyup.enter directive here -->
        <input type="password" class="form-control" id="password" v-model="credentials.password" required @keyup.enter="login">
      </div>
      <button type="submit" class="btn btn-primary mt-3">UPLOAD</button>
    </form>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'LoginComponent',
  data() {
    return {
      credentials: {
        username: '',
        password: ''
      }
    };
  },
  methods: {
    login() {
      axios.post(process.env.VUE_APP_BACKEND_URL + '/login', this.credentials)
        .then(response => {
          console.log('Login successful:', response);
          // Handle success (e.g., redirect, display a message)
        })
        .catch(error => {
          console.error('Login failed:', error);
          // Handle error (e.g., display an error message)
        });
    }
  }
}
</script>

<style scoped>
.login-container {
  max-width: 400px;
  margin: auto;
  padding-top: 50px;
}
</style>