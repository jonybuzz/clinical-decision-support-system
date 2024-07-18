// router/index.js
import Vue from 'vue';
import Router from 'vue-router';
import LoginComponent from '../components/LoginComponent.vue';
import UploadComponent from '../components/UploadComponent.vue';

Vue.use(Router);

export default new Router({
  routes: [
    {
      path: '/login',
      name: 'LoginComponent',
      component: LoginComponent
    },
    {
      path: '/success',
      name: 'UploadComponent',
      component: UploadComponent
    }
  ]
});