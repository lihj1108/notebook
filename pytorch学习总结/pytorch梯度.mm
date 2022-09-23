<map version="freeplane 1.9.13">
<!--To view this file, download free mind mapping software Freeplane from https://www.freeplane.org -->
<node TEXT="pytorch梯度" FOLDED="false" ID="ID_1090958577" CREATED="1606664858024" MODIFIED="1661239393387" VGAP_QUANTITY="3 pt">
<hook NAME="accessories/plugins/AutomaticLayout.properties" VALUE="ALL"/>
<hook NAME="MapStyle" background="#fbf1c7">
    <properties show_icon_for_attributes="true" edgeColorConfiguration="#808080ff,#ff0000ff,#0000ffff,#00ff00ff,#ff00ffff,#00ffffff,#7c0000ff,#00007cff,#007c00ff,#7c007cff,#007c7cff,#7c7c00ff" show_note_icons="true" fit_to_viewport="false" associatedTemplateLocation="template:/light_gruvbox_template.mm"/>

<map_styles>
<stylenode LOCALIZED_TEXT="styles.root_node" STYLE="oval" UNIFORM_SHAPE="true" VGAP_QUANTITY="24 pt">
<font SIZE="24"/>
<stylenode LOCALIZED_TEXT="styles.predefined" POSITION="right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="default" ID="ID_7974122" ICON_SIZE="12 pt" FORMAT_AS_HYPERLINK="false" COLOR="#3c3836" BACKGROUND_COLOR="#fbf1c7" STYLE="bubble" SHAPE_HORIZONTAL_MARGIN="8 pt" SHAPE_VERTICAL_MARGIN="5 pt" BORDER_WIDTH_LIKE_EDGE="false" BORDER_WIDTH="1.9 px" BORDER_COLOR_LIKE_EDGE="true" BORDER_COLOR="#f0f0f0" BORDER_DASH_LIKE_EDGE="true" BORDER_DASH="SOLID" VGAP_QUANTITY="3 pt">
<arrowlink SHAPE="CUBIC_CURVE" COLOR="#000000" WIDTH="2" TRANSPARENCY="200" DASH="" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_7974122" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<font NAME="SansSerif" SIZE="11" BOLD="false" STRIKETHROUGH="false" ITALIC="false"/>
<edge STYLE="bezier" COLOR="#93a1a1" WIDTH="3" DASH="SOLID"/>
<richcontent CONTENT-TYPE="plain/auto" TYPE="DETAILS"/>
<richcontent TYPE="NOTE" CONTENT-TYPE="plain/auto"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.details"/>
<stylenode LOCALIZED_TEXT="defaultstyle.attributes">
<font SIZE="9"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.note" COLOR="#000000" BACKGROUND_COLOR="#f6f9a1" TEXT_ALIGN="LEFT"/>
<stylenode LOCALIZED_TEXT="defaultstyle.floating">
<edge STYLE="hide_edge"/>
<cloud COLOR="#f0f0f0" SHAPE="ROUND_RECT"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.selection" COLOR="#ffffff" BACKGROUND_COLOR="#cc241d" BORDER_COLOR_LIKE_EDGE="false" BORDER_COLOR="#cc241d"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.user-defined" POSITION="right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="styles.important" ID="ID_103960811" BORDER_WIDTH="3 px" BORDER_COLOR_LIKE_EDGE="true" BORDER_COLOR="#f0f0f0">
<icon BUILTIN="yes"/>
<arrowlink COLOR="#cc241d" TRANSPARENCY="255" DESTINATION="ID_103960811"/>
<font SIZE="12" ITALIC="false"/>
<edge COLOR="#cc241d"/>
</stylenode>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.AutomaticLayout" POSITION="right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="AutomaticLayout.level.root" COLOR="#fdf6e3" BACKGROUND_COLOR="#282828" STYLE="bubble" SHAPE_HORIZONTAL_MARGIN="10 pt" SHAPE_VERTICAL_MARGIN="10 pt" BORDER_WIDTH="3.1 px" BORDER_COLOR_LIKE_EDGE="true" BORDER_COLOR="#2c2b29" BORDER_DASH_LIKE_EDGE="true">
<font NAME="Ubuntu" SIZE="18"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,1" COLOR="#282828" BACKGROUND_COLOR="#fbf1c7" STYLE="bubble" SHAPE_HORIZONTAL_MARGIN="8 pt" SHAPE_VERTICAL_MARGIN="5 pt" BORDER_COLOR="#2c2b29">
<font NAME="Ubuntu" SIZE="16"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,2" COLOR="#282828" BACKGROUND_COLOR="#ebdbb2" STYLE="bubble" SHAPE_HORIZONTAL_MARGIN="8 pt" SHAPE_VERTICAL_MARGIN="5 pt" BORDER_COLOR_LIKE_EDGE="true" BORDER_COLOR="#f0f0f0">
<font SIZE="14"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,3" COLOR="#282828" BACKGROUND_COLOR="#d79921" STYLE="bubble" SHAPE_HORIZONTAL_MARGIN="8 pt" SHAPE_VERTICAL_MARGIN="5 pt" BORDER_COLOR_LIKE_EDGE="true" BORDER_COLOR="#f0f0f0">
<font SIZE="12"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,4" COLOR="#ffffff" BACKGROUND_COLOR="#458588" BORDER_COLOR_LIKE_EDGE="true" BORDER_COLOR="#f0f0f0">
<font SIZE="11"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,5" COLOR="#ffffff" BACKGROUND_COLOR="#b16286" BORDER_COLOR_LIKE_EDGE="true" BORDER_COLOR="#f0f0f0">
<font SIZE="11"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,6" COLOR="#ffffff" BACKGROUND_COLOR="#689d6a" BORDER_COLOR_LIKE_EDGE="true" BORDER_COLOR="#f0f0f0">
<font SIZE="10"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,7" COLOR="#ffffff" BACKGROUND_COLOR="#a89984" BORDER_COLOR="#f0f0f0">
<font SIZE="10"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,8" BACKGROUND_COLOR="#ebdbb2" BORDER_COLOR="#f0f0f0">
<font SIZE="10"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,9" BORDER_COLOR="#f0f0f0" BACKGROUND_COLOR="#ebdbb2">
<font SIZE="10"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,10" BORDER_COLOR="#f0f0f0">
<font SIZE="9"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,11" BORDER_COLOR="#f0f0f0" BACKGROUND_COLOR="#98971a">
<font SIZE="9"/>
</stylenode>
</stylenode>
</stylenode>
</map_styles>
</hook>
<node TEXT="Autograd会自动创建一个有向无环图，记录所有执行的操作，它的叶子是输入张量，根是输出张量。使用链式法则从根到叶自动计算梯度。图在每次迭代时都是从头重新创建的。" POSITION="left" ID="ID_713513488" CREATED="1661236166822" MODIFIED="1661237469350"/>
<node TEXT="禁止计算梯度的方法:&#xa;requires_grad可以分别设置某个张量是否计算梯度&#xa;no_grad()和inference_mode()上下文中的所有张量都不会计算梯度；no_grad()会保留创建的张量，以备后续使用；inference_mode()中创建的张量在上下文结束后都会被销毁，所以推理速度更快" POSITION="right" ID="ID_1444886564" CREATED="1661237827512" MODIFIED="1661238060874">
<node TEXT="tensor.requires_grad=False" ID="ID_1214306340" CREATED="1661237838615" MODIFIED="1661237847510"/>
<node TEXT="torch.no_grad()" ID="ID_73473586" CREATED="1661237851543" MODIFIED="1661237865939"/>
<node TEXT="inference_mode()" ID="ID_591079019" CREATED="1661237870775" MODIFIED="1661237872276"/>
</node>
<node TEXT="model.train()和model.eval()的区别" POSITION="right" ID="ID_71610995" CREATED="1661238971704" MODIFIED="1661238989849">
<node TEXT="model.train()会启用batch normalization和dropout，用每一个batch数据的均值和方差对网络层中的数据进行标准化，随机另一部分的神经元失效。model.eval()不会启用batch normalization和dropout，用所有训练数据的均值和方差对网络层中的数据进行标准化，不进行随即舍弃神经元。" ID="ID_1922013149" CREATED="1661238990328" MODIFIED="1661239293051"/>
</node>
</node>
</map>
