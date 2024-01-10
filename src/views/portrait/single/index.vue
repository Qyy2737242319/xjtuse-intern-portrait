<template>
  <div class="centered-container">
    <div class="title">用户个体画像查询</div>
    <!-- <el-form :model="numberValidateForm" label-width="80px" class="demo-ruleForm">
      <el-form-item
        label="用户查询"
        prop="id"
        placeholder="请输入用户id或者手机号查询："
        :rules="[
          { required: false, message: '查询条件不能为空' },
          { type: 'number', message: '请输入有效的id或者手机号' },
        ]"
        class="custom-input"
      >
        <el-input
          v-model.number="numberValidateForm.age"
          type="text"
          autocomplete="off"
          placeholder="请输入用户id或者手机号查询："
        />
      </el-form-item> -->
    <el-input
      v-model="inputValue.id"
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
  <div ref="chartContainer" style="height: 800px; width: 100%"></div>
</template>

<script setup lang="ts">
import * as echarts from 'echarts'
import { reactive, ref, onMounted } from 'vue'
import type { FormInstance } from 'element-plus'
import { useRouter, useRoute } from 'vue-router'
import { Loading } from '@element-plus/icons-vue/dist/types';
let $router = useRouter()
let $route = useRoute()
// eslint-disable-next-line no-redeclare
// type FormInstance = InstanceType<typeof ElForm>
// const formRef = ref<FormInstance>()
// const numberValidateForm = ref({ id: '' })
// const submitForm = () => {
//   const id = numberValidateForm.value.id
//   if (id === '1') {
//     $router.push('/portrait/base/user1')
//   } else if (id === '13361629658') {
//     $router.push('/portrait/base/user2')
//   } else {
//     $router.push('/portrait/base/user3')
//   }
// }
// const resetForm = () => {
//   numberValidateForm.value.id = ''
// }
const inputValue = reactive({ id: ' ' })
const submitForm = () => {
  const id = inputValue.id
  if (id === '1') {
    $router.push('/portrait/base/user1')
  } else if (id === '13361629658') {
    $router.push('/portrait/base/user2')
  } else {
    $router.push('/portrait/base/user3')
  }
}
const resetForm = () => {
  inputValue.id = ''
}
const baseName = '用户画像'
const chartData = {
  人口属性: ['男', '90后', '中国'],
  商业属性: ['有券必买', '高价值用户', '消费频率高'],
  行为属性: ['购物频率高', '天天浏览', '下单时间快'],
}

const datas = [{ name: baseName || '', draggable: true }]
const lines: { source: number; target: number; value: string }[] = []
let categoryIdx = 0
let keyIndex = 0
let dataIndex = 0

for (const [key, values] of Object.entries(chartData)) {
  keyIndex = dataIndex;
  datas.push({ name: key, category: categoryIdx, draggable: true })
  keyIndex++
  dataIndex++
  lines.push({
    source: 0,
    target: keyIndex,
    value: '',
  })

  values.forEach((val) => {
    datas.push({ name: val, category: categoryIdx, draggable: true })
    dataIndex++;
    lines.push({
      source: keyIndex,
      target: dataIndex,
      value: '',
    })
  })

  categoryIdx++
}

const option = {
  title: {
    text: '',
  },
  tooltip: {},
  animationDurationUpdate: 1500,
  label: {
    normal: {
      show: true,
      textStyle: {
        fontSize: 12,
      },
    },
  },
  series: [
    {
      type: 'graph',
      layout: 'force',
      symbolSize: 45,
      legendHoverLink: true,
      focusNodeAdjacency: true,
      roam: true,
      label: {
        normal: {
          show: true,
          position: 'inside',
          textStyle: {
            fontSize: 12,
          },
        },
      },
      force: {
        repulsion: 1000,
      },
      edgeSymbolSize: [4, 50],
      edgeLabel: {
        normal: {
          show: true,
          textStyle: {
            fontSize: 10,
          },
          formatter: '{c}',
        },
      },
      categories: [
        {
          itemStyle: {
            normal: {
              color: '#BB8FCE',
            },
          },
        },
        {
          itemStyle: {
            normal: {
              color: '#0099FF',
            },
          },
        },
        {
          itemStyle: {
            normal: {
              color: '#5DADE2',
            },
          },
        },
      ],
      data: datas,
      links: lines,
      lineStyle: {
        normal: {
          opacity: 0.9,
          width: 1,
          curveness: 0,
        },
      },
    },
  ],
}

const chartContainer = ref()

onMounted(() => {
  const container = chartContainer.value
  const myChart = echarts.init(container)
  myChart.setOption(option)
})
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
