<template>
  <div class="centered-container">
    <div class="title">用户最喜爱的十件商品推荐</div>
    <el-input
      v-model="inputValue"
      type="text"
      placeholder="请输入用户id或者手机号查询："
      clearable
      class="custom-input"
    />
    <el-form-item class="centered-buttons">
      <el-button type="primary" @click="submitForm">查询</el-button>
      <el-button @click="resetForm">重置</el-button>
    </el-form-item>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
let $router = useRouter()
const inputValue = ref({})
const submitForm = () => {
  const id = inputValue.value
  if (id === 1) {
    $router.push('/portrait/base/user1')
  } else if (id === 13361629658) {
    $router.push('/portrait/base/user2')
  } else {
    $router.push('/portrait/base/user3')
  }
}
const resetForm = () => {
  inputValue.value = ''
}
</script>

<style scoped>
.custom-input .el-input {
  width: 150px; /* Set the desired width */
  margin: 0 auto; /* Center the input field */
}
.centered-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
}

.title {
  font-size: 30px;
  margin-bottom: 20px;
}

.custom-input .el-input {
  width: 300px;
}

.centered-buttons {
  text-align: center;
  margin-top: 20px;
}

.el-button {
  margin-right: 10px;
}
</style>
