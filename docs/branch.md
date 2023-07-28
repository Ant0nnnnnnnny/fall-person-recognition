# github开发规范

考虑到大创项目可能会采用多人协作的方式，为了方便版本与代码的统一管理，使用 github 作为代码存储的首要工具。

## 分支使用说明

首先，在 github 中代码存在分支(` branch `)的概念，在每个分支可以存在不同的代码更改。 分支的存在可以保证每个人在修改项目代码的同时不影响到现有的代码。在自己的分支开发完成后，可以申请合并(`pull request`, `merge`)到主分支。这里简单说一下分支的使用规范。

### main / master 分支

`main`/ `master` 分支**必须**永远是可用的稳定版本。不可在`master`分支上进行开发。在` master `分支基础上添加新分支后，必须**完成对应的开发后**提交`pull request`, 通过代码审核后合并到master分支。

### develop, feature 和hotfix分支

---

+ `develop`: 开发主分支，所有的新功能以 `develop` 分支为基础创建新的分支进行开发，该分支只做合并操作, 不可在 `develop` 分支上进行开发。
  
+ `feature-xxx`: 功能开发分支。该分支为在 `develop` 基础上创建的分支, 以自己开发的对应功能模块命名, 测试正常后合并到develop分支。

+ `feature-xxx-fix`: 功能bug修复分支。如果`feature`合并后发现bug, 则在`develop`的基础上创建本分支，完成修复后合并到`develop`分支。

+ `hotfix-xxx`: 紧急bug修改分支。 在`master`上创建，修复完成后合并至`master`。一般用不上。

---
