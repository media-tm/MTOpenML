# 深度学习-82:OpenCV与深度学习

> [深度学习原理与实践(开源图书)-总目录](https://blog.csdn.net/shareviews/article/details/83040730)

随着机器学习，计算机视觉和计算能力的日益成熟，计算机视觉被广泛应用于人机互动、物体识别、图像分割、人脸识别、动作识别、运动跟踪、机器人、运动分析、机器视觉、结构分析和汽车安全驾驶等。计算机视觉伴随着比较多的推理型和场景分析型的视觉任务，大量机器学习算法被引入计算机视觉；最近10年，神经网络和深度学习发展迅猛，也被纳入了计算机视觉分析的工具箱。本节以应用最广的OpenCV框架，宏观观察一下人工智能(AI)给计算机视觉带来的革命性变化。

## OpenCV的初衷和愿景

计算机视觉是在图像处理的基础上发展起来的新兴学科，一门由信息处理、计算机、机器人、人工智能、图像处理和认知神经科学等融合的学问。OpenCV是一个开源的计算机视觉库，由英特尔公司资助开发并开源。OpenCV不是唯一的开源计算机视觉库，但它绝对是最好的。

计算机视觉市场巨大，各种场景非常复杂，但是计算机视觉没有标准的API, 开发各类计算机视觉应用异常困难。由于没有标准API,代码质量和代码兼容性堪忧，交叉验证想法和论文，几乎是不可能的事情；商业化项目成本难以控制，技术门槛非常高，难以快速原型开发；计算机视觉多以嵌入式外设(视频监控，制造控制系统，医疗设备)作为数据输入，硬件差异造成通用解决方案的架构异常复杂; 新算法和新想法，难以快速验证和产品化。基于以上显著的问题，OpenCV致力于提供简化和稳定的计算机视觉解决方案，这构成了OpenCV的最核心的初衷和愿景。
计算机视觉库OpenCV具有以下特性：

- 拥有丰富的图像处理和计算机视觉领域通用的算法；
- 支持机器学习和深度学习做视觉分析，机器学习库侧重于统计方面的模式识别和聚类(clustering)，深度学习库侧重视觉任务；
- 支持硬件加速提高计算性能；
- 拥有500 多个C函数的跨平台的高级 API；
- 支持跨平台和多种编程语言绑定
- 覆盖了计算机视觉的许多应用领域，如工厂产品检测、医学成像、信息安全、用户界面、摄像机标定、立体视觉和机器人等。

## OpenCV的应用

- OpenCV 可用于视频监控领域
- OpenCV 可用于游戏开发领域
- OpenCV 可用于地理图像的计算机定标和图像拼接领域
- OpenCV 可用于安全监控、生物医学分析领域
- OpenCV 可用于工业控制，生产自动化领域
- OpenCV 可用于无人飞行器，无人汽车和无人水下机器人领域
- OpenCV 可用于智能机器人领域
- OpenCV 可用于模式识别、三维重建领域
- OpenCV 可用于物体跟踪领域

## opencv 版本历史

- 2010年12月06日，OpenCV 2.2 版本发布。
- 2013年07月03日，OpenCV 2.4 版本发布。开始支持Android和各种深度学习算法。
- 2015年06月04日，OpenCV 3.0 版本发布。
- 2015年12月21日，OpenCV 3.1 版本发布。
- 2016年12月23日，OpenCV 3.2 版本发布。
- 2017年08月03日，OpenCV 3.3 版本发布。
- 2018年07月04日，OpenCV 3.4 版本发布。开始支持人类深度学习算法，深度学习算法效果赶超前辈。

## 系列文章

- [深度学习原理与实践(开源图书)-总目录](https://blog.csdn.net/shareviews/article/details/83040730)
- [机器学习原理与实践(开源图书)-总目录](https://blog.csdn.net/shareviews/article/details/83030331)
- [Github: 机器学习&深度学习理论与实践(开源图书)](https://github.com/media-tm/MTOpenML)

## 参考文献

- [1] Ian Goodfellow, Yoshua Bengio. [Deep Learning](http://www.deeplearningbook.org/). MIT Press. 2016.
- [2] 焦李成等. 深度学习、优化与识别. 清华大学出版社. 2017.
- [3] 佩德罗·多明戈斯. 终极算法-机器学习和人工智能如何重塑世界. 中信出版社. 2018.
- [4] 雷.库兹韦尔. 人工智能的未来-揭示人类思维的奥秘.  浙江人民出版社. 2016.