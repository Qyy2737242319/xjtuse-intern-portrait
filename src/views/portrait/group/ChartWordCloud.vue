<!-- <template>
  <div id="charts-content"></div>
</template>

<script setup lang="ts">
import * as echarts from 'echarts'
import 'echarts-wordcloud'
import { onMounted } from 'vue'
// 引入 lodash 中的 merge、深克隆
import merge from 'lodash/merge'
 
 
const props = withDefaults(
	defineProps<{
		options: any
	}>(),
	{},
)
 
 
// 词云图默认属性
const defaultSeries = [{
	type: 'wordCloud',

	shape: 'circle',

	keepAspect: false,

	left: 'center',
	top: 'center',
	width: '100%',
	height: '100%',
	right: null,
	bottom: null,
	// 词云文本大小范围,  默认为最小12像素，最大60像素
	sizeRange: [12, 60],
	// 词云文字旋转范围和步长。 文本将通过旋转在[-90，90]范围内随机旋转步骤45
	// 如果都设置为 0 , 则是水平显示
	rotationRange: [-90, 90],
	rotationStep: 45,
	/**
	 * 词间距, 距离越大，单词之间的间距越大, 单位像素
	 * 这里间距太小的话，会出现大词把小词套住的情况，比如一个大的口字，中间会有比较大的空隙，这时候他会把一些很小的字放在口字里面，这样的话，鼠标就无法选中里面的那个小字
	 */
	gridSize: 8,
	// 设置为true可以使单词部分在画布之外绘制, 允许绘制大于画布大小的单词
	drawOutOfBound: false,
	/**
	 * 布局的时候是否有动画
	 * 注意：禁用时，当单词较多时，将导致UI阻塞。
	 */
	layoutAnimation: true,
	// 这是全局的文字样式，相对应的还可以对每个词设置字体样式
	textStyle: {
		fontFamily: 'sans-serif',
		fontWeight: 'bold',
		// 颜色可以用一个函数来返回字符串
		color: function () {
          // 随机颜色
          return (
            'rgb(' +
            [
              Math.round(Math.random() * 160),
              Math.round(Math.random() * 160),
              Math.round(Math.random() * 160),
            ].join(',') +
            ')'
          )
        },
	},
	// 鼠标hover的特效样式
	emphasis: {
		focus: 'self',
		textStyle: {
			textShadowBlur: 10,
			textShadowColor: '#999'
		}
	},
 
	/**
	 * 词云数据，必须是一个数组，每个数组项必须有name和value属性
	 * 设置单个文本的样式：  textStyle 
	 * 
	 * 例：{
					name: '',
					value: 40,
					textStyle: {
					}
				},
	 */
	data: []
}]
let seriesData = props.options.series;
const series = merge({}, defaultSeries[0], seriesData[0]) // {}表示合并后的新对象，可以传入一个空对象作为初始值
 
function DrawWordCloud() {
	// 词云
	let mychart = echarts.init(document.getElementById("charts-content")) // 可以设置主题色'dark'
	mychart.setOption({
		series: series
	})
}
 
onMounted(() => {
	DrawWordCloud()
})
</script>
 
<style scoped lang="scss"></style> -->
