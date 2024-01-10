/*
 * @Description: Stay hungry，Stay foolish
 * @Author: Huccct
 * @Date: 2023-05-19 17:46:49
 * @LastEditors: Huccct
 * @LastEditTime: 2023-06-02 10:33:35
 */
export const constantRoute = [
  {
    path: '/login',
    component: () => import('@/views/login/index.vue'),
    name: 'login',
    meta: {
      title: 'login',
      hidden: true,
    },
  },
  {
    path: '/',
    component: () => import('@/layout/index.vue'),
    name: 'layout',
    meta: {
      title: '',
      hidden: false,
      icon: '',
    },
    redirect: '/home',
    children: [
      {
        path: '/home',
        component: () => import('@/views/home/index.vue'),
        meta: {
          title: '首页',
          hidden: false,
          icon: 'HomeFilled',
        },
      },
    ],
  },
  {
    path: '/screen',
    component: () => import('@/views/screen/index.vue'),
    name: 'Screen',
    meta: {
      title: '数据大屏',
      hidden: false,
      icon: 'Platform',
    },
  },
  {
    path: '/query',
    component: () => import('@/layout/index.vue'),
    name: 'Query',
    meta: {
      title: '用户标签查询',
      hidden: false,
      icon: 'Search',
    },
    redirect: '/query/base',
    children: [
      {
        path: '/query/base',
        component: () => import('@/views/query/base/index.vue'),
        name: 'Base',
        meta: {
          title: '用户基础标签查询',
          hidden: false,
          icon: 'User',
        },
      },
      {
        path: '/query/union',
        component: () => import('@/views/query/union/index.vue'),
        name: 'Union',
        meta: {
          title: '用户组合标签查询',
          hidden: false,
          icon: 'Avatar',
        },
      },
    ],
  },
  {
    path: '/portrait',
    component: () => import('@/layout/index.vue'),
    name: 'Portrait',
    meta: {
      title: '用户画像查询',
      hidden: false,
      icon: 'Search',
    },
    redirect: '/portrait/base',
    children: [
      {
        path: '/portrait/base',
        component: () => import('@/views/portrait/single/index.vue'),
        name: 'Single',
        meta: {
          title: '用户个体画像查询',
          hidden: false,
          icon: 'User',
        },
      },
      {
        path: '/portrait/base/user1',
        component: () => import('@/views/portrait/single/peo1.vue'),
        name: 'User',
        meta: {
          title: '用户个体画像',
          hidden: true,
          icon: 'Platform',
        },
      },
      {
        path: '/portrait/base/user2',
        component: () => import('@/views/portrait/single/peo3.vue'),
        name: 'User1',
        meta: {
          title: '用户个体画像',
          hidden: true,
          icon: 'Platform',
        },
      },
      {
        path: '/portrait/base/user3',
        component: () => import('@/views/portrait/single/peo3.vue'),
        name: 'User2',
        meta: {
          title: '用户个体画像',
          hidden: true,
          icon: 'Platform',
        },
      },
      {
        path: '/portrait/group',
        component: () => import('@/views/portrait/group/index.vue'),
        name: 'Group',
        meta: {
          title: '用户群体画像查询',
          hidden: false,
          icon: 'Avatar',
        },
      },
    ],
  },
  {
    path: '/ml',
    component: () => import('@/layout/index.vue'),
    name: 'Ml',
    meta: {
      title: '大数据应用',
      hidden: false,
      icon: 'DataLine',
    },
    redirect: '/ml/sort',
    children: [
      {
        path: '/ml/sort',
        component: () => import('@/views/ml/sort/index.vue'),
        name: 'Sort',
        meta: {
          title: '用户价值分类',
          hidden: false,
          icon: 'User',
        },
      },
      {
        path: '/ml/search',
        component: () => import('@/views/ml/search/index.vue'),
        name: 'Search',
        meta: {
          title: '用户价值查询',
          hidden: false,
          icon: 'Avatar',
        },
      },
      {
        path: '/ml/recommend',
        component: () => import('@/views/ml/recommend/index.vue'),
        name: 'Recommend',
        meta: {
          title: '用户商品推荐',
          hidden: false,
          icon: 'Avatar',
        },
      },
    ],
  },
  {
    path: '/404',
    component: () => import('@/views/404/index.vue'),
    name: '404',
    meta: {
      title: '404',
      hidden: true,
    },
  },
]

export const asyncRoute = [
  /* {
    path: '/query',
    component: () => import('@/layout/index.vue'),
    name: 'Query',
    meta: {
      title: '用户标签查询',
      hidden: false,
      icon: 'Search',
    },
    redirect: '/query/base',
    children: [
      {
        path: '/query/base',
        component: () => import('@/views/query/base/index.vue'),
        name: 'Base',
        meta: {
          title: '用户基础标签查询',
          hidden: false,
          icon: 'User',
        },
      },
      {
        path: '/query/union',
        component: () => import('@/views/query/union/index.vue'),
        name: 'Union',
        meta: {
          title: '用户组合标签查询',
          hidden: false,
          icon: 'Avatar',
        },
      },
    ],
  },
  {
    path: '/portrait',
    component: () => import('@/layout/index.vue'),
    name: 'Portrait',
    meta: {
      title: '用户画像查询',
      hidden: false,
      icon: 'Search',
    },
    redirect: '/portrait/base',
    children: [
      {
        path: '/portrait/base',
        component: () => import('@/views/portrait/single/index.vue'),
        name: 'Single',
        meta: {
          title: '用户个体画像查询',
          hidden: false,
          icon: 'User',
        },
      },
      {
        path: '/portrait/group',
        component: () => import('@/views/portrait/group/index.vue'),
        name: 'Group',
        meta: {
          title: '用户群体画像查询',
          hidden: false,
          icon: 'Avatar',
        },
      },
    ],
  },
  {
    path: '/ml',
    component: () => import('@/layout/index.vue'),
    name: 'Ml',
    meta: {
      title: '大数据应用',
      hidden: false,
      icon: 'Search',
    },
    redirect: '/ml/sort',
    children: [
      {
        path: '/ml/sort',
        component: () => import('@/views/ml/sort/index.vue'),
        name: 'Sort',
        meta: {
          title: '用户价值分类',
          hidden: false,
          icon: 'User',
        },
      },
      {
        path: '/ml/search',
        component: () => import('@/views/ml/search/index.vue'),
        name: 'Search',
        meta: {
          title: '用户价值查询',
          hidden: false,
          icon: 'Avatar',
        },
      },
      {
        path: '/ml/recommend',
        component: () => import('@/views/ml/recommend/index.vue'),
        name: 'Recommend',
        meta: {
          title: '用户商品推荐',
          hidden: false,
          icon: 'Avatar',
        },
      },
    ],
  }, */
  {
    path: '/acl',
    component: () => import('@/layout/index.vue'),
    name: 'Acl',
    meta: {
      title: '权限管理',
      hidden: false,
      icon: 'Lock',
    },
    redirect: '/acl/user',
    children: [
      {
        path: '/acl/user',
        component: () => import('@/views/acl/user/index.vue'),
        name: 'User',
        meta: {
          title: '用户管理',
          hidden: false,
          icon: 'User',
        },
      },
      {
        path: '/acl/role',
        component: () => import('@/views/acl/role/index.vue'),
        name: 'Role',
        meta: {
          title: '角色管理',
          hidden: false,
          icon: 'Avatar',
        },
      },
      {
        path: '/acl/permission',
        component: () => import('@/views/acl/permission/index.vue'),
        name: 'Permission',
        meta: {
          title: '菜单管理',
          hidden: false,
          icon: 'List',
        },
      },
    ],
  },
  {
    path: '/product',
    component: () => import('@/layout/index.vue'),
    name: 'Product',
    meta: {
      title: '商品管理',
      hidden: false,
      icon: 'Goods',
    },
    redirect: '/product/trademark',
    children: [
      {
        path: '/product/trademark',
        component: () => import('@/views/product/trademark/index.vue'),
        name: 'Trademark',
        meta: {
          title: '品牌管理',
          icon: 'ShoppingCart',
          hidden: false,
        },
      },
      {
        path: '/product/attr',
        component: () => import('@/views/product/attr/index.vue'),
        name: 'Attr',
        meta: {
          title: '属性管理',
          icon: 'Management',
          hidden: false,
        },
      },
      {
        path: '/product/spu',
        component: () => import('@/views/product/spu/index.vue'),
        name: 'Spu',
        meta: {
          title: 'Spu',
          icon: 'SetUp',
          hidden: false,
        },
      },
      {
        path: '/product/sku',
        component: () => import('@/views/product/sku/index.vue'),
        name: 'Sku',
        meta: {
          title: 'Sku',
          icon: 'ScaleToOriginal',
          hidden: false,
        },
      },
    ],
  },
]

export const anyRoute = {
  path: '/:pathMatch(.*)*',
  redirect: '/404',
  name: 'Any',
  meta: {
    title: '任意路由',
    hidden: true,
  },
}
