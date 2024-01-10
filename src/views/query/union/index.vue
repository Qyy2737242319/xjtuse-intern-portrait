<template>
  <div>
    <el-form :inline="true" :model="formInline" class="demo-form-inline">
      <el-form-item label="请输入组合标签名称：">
        <el-select v-model="formInline.region" placeholder="请选择：" clearable>
          <el-option label="人口属性" value="p" />
          <el-option label="商业属性" value="c" />
          <el-option label="行为属性" value="a" />
          <el-option label="用户价值" value="v" />
        </el-select>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="onSubmit">查询</el-button>
      </el-form-item>
    </el-form>
    <el-divider />
  </div>
  <div class="search-container">
    <span class="search-text">请输入关键词进行搜索：</span>
    <el-input
      v-model="search"
      size="small"
      placeholder="请输入关键词"
      :style="{ width: '200px', height: '30px' }"
    />
  </div>
  <el-table :data="filterTableData" border style="width: 100%">
    <el-table-column label="标签级别" prop="pid" />
    <el-table-column label="标签名" prop="name" />
    <el-table-column label="标签人数" prop="number" />
    <el-table-column label="标签占比" prop="percent" />
    <el-table-column label="标签状态" prop="status">
      <template #default="{ row }">
        <el-switch
          v-model="value3"
          inline-prompt
          active-text="已启用"
          inactive-text="已禁用"
          :active-icon-class="row.switchStatus ? 'el-icon-check' : ''"
          :inactive-icon-class="row.switchStatus ? '' : 'el-icon-close'"
        />
      </template>
    </el-table-column>
    <el-table-column label="标签创建时间" prop="date" />
    <el-table-column label="标签修改时间" prop="change" />
    <el-table-column align="left" label="操作" prop="opr">
      <template #default="scope">
        <el-button size="small" @click="handleEdit(scope.$index, scope.row)">编辑</el-button>
        <el-button
          size="small"
          type="danger"
          @click="handleDelete(scope.$index, scope.row)"
        >
          删除
        </el-button>
      </template>
    </el-table-column>
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
const value3 = ref(true)
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
  pid: number
  name: string
  date: string
  change: string
  number: string
  percent: string
  status: string
  hide: string
}
const search = ref('')
/* const filterTableData = computed(() =>
  tableData.filter(
    (data) =>
      !search.value ||
      data.name.toLowerCase().includes(search.value.toLowerCase()),
  ),
) */
const handleEdit = (index: number, row: User) => {
  console.log(index, row)
}
const handleDelete = (index: number, row: User) => {
  console.log(index, row)
}

const tableData: User[] = [
  {
    pid: 4,
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
