<template>
  <div class="login-container">
    <img alt="Vue logo" src="../assets/logo.png">
    <form @submit.prevent="login">
      <div class="form-group mb-3">
        <label for="username">Username</label>
        <input type="text" class="form-control" id="username" v-model="credentials.username" required>
      </div>
      <div class="form-group mb-3">
        <label for="password">Password</label>
        <input type="password" class="form-control" id="password" v-model="credentials.password" required
               @keyup.enter="login">
      </div>
      <!-- Display error message if it exists -->
      <p v-if="errorMessage" class="text-danger">{{ errorMessage }}</p>
      <button type="submit" class="btn btn-primary mt-3">Login</button>
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
      },
      errorMessage: '' // Added errorMessage data property
    };
  },
  methods: {
    login() {
      axios.post(process.env.VUE_APP_BACKEND_URL + '/login', this.credentials)
          .then(response => {
            console.log('Login successful:', response);
            // Redirect to UploadComponent
            const targetRoute = {name: 'UploadComponent'};
            console.log("KE:" + this.$router.currentRoute.name)
            // Check if the current route is the same as the target route
            if (this.$router.currentRoute.name !== targetRoute.name) {
              this.$router.push(targetRoute);
            }
          })
          .catch(error => {
            console.error('Login failed:', error);
            this.errorMessage = 'Login failed. Please check your credentials and try again.';
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

.text-danger {
  color: #dc3545; /* Bootstrap danger color */
}
</style>