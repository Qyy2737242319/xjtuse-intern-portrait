<template>
  <div>
    <h1 align="center">黄艺的用户画像</h1>
    <div ref="chartContainer" style="height: 800px; width: 100%"></div>
  </div>
</template>
<script setup lang="ts">
import * as echarts from 'echarts'
import { ref, onMounted } from 'vue'
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
  keyIndex = dataIndex
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
    dataIndex++
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
<style scoped></style>
