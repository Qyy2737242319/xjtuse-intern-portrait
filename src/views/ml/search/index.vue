<template>
  <div>
    <el-form :inline="true" :model="formInline" class="demo-form-inline">
      <el-form-item label="请输入用户价值类型：">
        <el-select v-model="formInline.region" placeholder="请选择：" clearable>
          <el-option label="重点发展用户" value="p" />
          <el-option label="高价值用户" value="c" />
          <el-option label="重点挽留客户" value="a" />
          <el-option label="一般挽留用户" value="v" />
          <el-option label="一般发展用户" value="v" />
          <el-option label="易流失用户" value="v" />
        </el-select>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="onSubmit">查询</el-button>
      </el-form-item>
    </el-form>
    <el-divider />
  </div>
  <!--   <div class="search-container">
    <span class="search-text">请输入关键词进行搜索：</span>
    <el-input
      v-model="search"
      size="small"
      placeholder="请输入关键词"
      :style="{ width: '200px', height: '30px' }"
    />
  </div> -->
  <el-table :data="filterTableData" border style="width: 100%">
    <el-table-column label="用户id" prop="id" />
    <el-table-column label="用户姓名" prop="name" />
    <el-table-column label="用户类型" prop="are" />
    <el-table-column label="用户单笔消费最高" prop="buy" />
    <!-- <el-table-column label="标签状态" prop="status"> -->
    <el-table-column label="用户消费周期" prop="dec" />
    <el-table-column label="最近一次消费" prop="time" />
    <el-table-column label="消费频率" prop="frequent" />
    <el-table-column label="消费总金额" prop="total" />
  </el-table>
  <el-divider />
  <div class="page-container">
    <!-- 页面内容 -->

    <div class="demo-pagination-block" align="center">
      <el-pagination
        :page-sizes="[10, 20, 30, 40]"
        :background="true"
        layout="total, sizes, prev, pager, next, jumper"
        :total="40"
        @size-change="handleSizeChange"
        @current-change="handleCurrentChange"
        v-model:current-page="currentPage4"
        v-model:page-size="pageSize4"
        align="center"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, reactive } from 'vue'
const currentPage4 = ref(4)
const pageSize4 = ref(10)
// eslint-disable-next-line vue/no-setup-props-destructure
const handleSizeChange = (val: number) => {
  console.log(`${val} items per page`)
}
const handleCurrentChange = (val: number) => {
  console.log(`current page: ${val}`)
}
const formInline = reactive({
  region: '',
})
const filterTableData = computed(() =>
  tableData.filter(
    (data) =>
      (!search.value ||
        data.name.toLowerCase().includes(search.value.toLowerCase())) &&
      (!formInline.region || data.hide === formInline.region),
  ),
)
const onSubmit = () => {
  console.log('submit!')
}
interface User {
  id: number
  name: string
  are: string
  buy: string
  dec: string
  time: string
  requent: string
  total: string
}
const search = ref('')
/* const filterTableData = computed(() =>
  tableData.filter(
    (data) =>
      !search.value ||
      data.name.toLowerCase().includes(search.value.toLowerCase()),
  ),
) */

const tableData: User[] = [
  {
    id: 1,
    date: '2023-12-30',
    name: '高净值用户',
    change: '2024-1-03',
    number: '480',
    percent: '48%',
    status: '已启用',
    hide: 'p',
  },
]
</script>

<style scoped>
.demo-form-inline .el-input {
  --el-input-width: 220px;
}
.page-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  min-height: 100vh;
  padding-bottom: 20px; /* 调整底部空白 */
}
.search-container {
  display: flex;
  align-items: center;
}

.search-text {
  margin-right: 10px;
  font-size: 16px;
  color: blue; /* 调整颜色 */
}
.el-icon-check {
  color: green;
}

.el-icon-close {
  color: red;
}
</style>
