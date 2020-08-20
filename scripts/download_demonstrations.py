import minerl
import shutil

# Download world
minerl.data.download('./res/',experiment='MineRLNavigate-v0')
minerl.data.download('./res/',experiment='MineRLNavigateDense-v0')

# Delete download folder
shutil.rmtree('./res/download', ignore_errors=True)