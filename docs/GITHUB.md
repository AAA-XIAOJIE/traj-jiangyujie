# 本地 Git 与 GitHub

仓库只收录这门课的代码、配置、图和结果，按 `week01` … `week08` 组织。原始下载、虚拟环境、缓存与本机操作脚本由 `.gitignore` 排除。

## 后续每周更新

```bash
git status
git add week02 src configs tests
git diff --cached --stat
git commit -m "feat(week02): add individual kinematics analysis"
git push
```

把命令中的 `week02` 与提交说明替换为实际周次。每周至少放代表图、结论和复现入口；不要用空目录或占位文字冒充已完成作业。

## 首次发布（若远端尚未建立）

在自己的 GitHub 账号创建空仓库 `traj-jiangyujie`，不额外初始化 README、License 或 .gitignore；本地已经提供对应文件。课程链接需要老师可访问，可选公开仓库，或在私有仓库里授予老师访问权。

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/traj-jiangyujie.git
git push -u origin main
```

使用 Git Credential Manager 或 GitHub CLI 的浏览器授权流程登录。账号密码和访问令牌不要写入代码、配置或提交历史。远端创建/推送是否已完成，以实际远端地址和 `git ls-remote origin` 为准。
