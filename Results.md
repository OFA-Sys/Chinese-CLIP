**MUGE Text-to-Image Retrieval (Official Validation Set)**:
<table border="1" width="100%">
    <tr align="center">
        <th>Setup</th><th colspan="4">Zero-shot</th><th colspan="4">Finetune</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1</td><td>R@5</td><td>R@10</td><td>MR</td><td>R@1</td><td>R@5</td><td>R@10</td><td>MR</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>RN50</sub></td><td>42.6</td><td>68.6</td><td>77.9</td><td>63.0</td><td>48.6</td><td>75.1</td><td>84.0</td><td>69.2</td>
    </tr>  
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B/16</sub></td><td>52.1</td><td>76.7</td><td>84.4</td><td>71.1</td><td>58.4</td><td>83.6</td><td>90.0</td><td>77.4</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14</sub></td><td>56.3</td><td>79.8</td><td>86.2</td><td>74.1</td><td>63.3</td><td>85.6</td><td>91.3</td><td>80.1</td>
    </tr>  
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14@336px</sub></td><td>59.0</td><td>81.4</td><td>87.8</td><td>76.1</td><td>65.3</td><td>86.7</td><td>92.1</td><td>81.3</td>
    </tr>    
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-H/14</sub></td><td><b>63.0</b></td><td><b>84.1</b></td><td><b>89.2</b></td><td><b>78.8</b></td><td><b>68.9</b></td><td><b>88.7</b></td><td><b>93.1</b></td><td><b>83.6</b></td>
    </tr>    
</table>
<br>

**Flickr30K-CN Retrieval (Official Test Set)**:
<table border="1" width="120%">
	<tr align="center">
        <th>Task</th><th colspan="6">Text-to-Image</th><th colspan="6">Image-to-Text</th>
    </tr>
    <tr align="center">
        <th>Setup</th><th colspan="3">Zero-shot</th><th colspan="3">Finetune</th><th colspan="3">Zero-shot</th><th colspan="3">Finetune</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>RN50</sub></td><td>48.8</td><td>76.0</td><td>84.6</td><td>66.7</td><td>89.4</td><td>94.1</td><td>60.0</td><td>85.9</td><td>92.0</td><td>84.2</td><td>96.7</td><td>98.0</td>
    </tr>  
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B/16</sub></td><td>62.7</td><td>86.9</td><td>92.8</td><td>79.1</td><td>94.8</td><td>97.4</td><td>74.6</td><td>93.5</td><td>97.1</td><td>93.5</td><td>99.0</td><td>99.5</td>
    </tr>  
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14</sub></td><td>68.0</td><td>89.7</td><td>94.4</td><td>82.7</td><td>96.7</td><td>98.6</td><td>80.2</td><td>96.6</td><td>98.2</td><td>96.1</td><td>99.5</td><td>99.9</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14@336px</sub></td><td>69.0</td><td>90.7</td><td>95.4</td><td><b>84.4</b></td><td><b>97.1</b></td><td><b>98.7</b></td><td><b>83.3</b></td><td>97.2</td><td>98.5</td><td><b>96.6</b></td><td><b>99.8</b></td><td><b>100.0</b></td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-H/14</sub></td><td><b>71.2</b></td><td><b>91.4</b></td><td><b>95.5</b></td><td>83.8</td><td>96.9</td><td>98.6</td><td>81.6</td><td><b>97.5</b></td><td><b>98.8</b></td><td>95.3</td><td>99.7</td><td><b>100.0</b></td>
    </tr>  
</table>
<br>

**COCO-CN Retrieval (Official Test Set)**:
<table border="1" width="120%">
	<tr align="center">
        <th>Task</th><th colspan="6">Text-to-Image</th><th colspan="6">Image-to-Text</th>
    </tr>
    <tr align="center">
        <th>Setup</th><th colspan="3">Zero-shot</th><th colspan="3">Finetune</th><th colspan="3">Zero-shot</th><th colspan="3">Finetune</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>RN50</sub></td><td>48.1</td><td>81.3</td><td>90.5</td><td>66.8</td><td>91.1</td><td>97.0</td><td>51.6</td><td>81.2</td><td>90.5</td><td>68.4</td><td>93.3</td><td>97.8</td>
    </tr>  
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B/16</sub></td><td>62.2</td><td>86.6</td><td>94.9</td><td>77.0</td><td>97.1</td><td>99.0</td><td>57.0</td><td>84.1</td><td>93.6</td><td>77.4</td><td>96.2</td><td>98.9</td>
    </tr>  
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14</sub></td><td>64.0</td><td>89.2</td><td>94.4</td><td>78.9</td><td>96.3</td><td>99.0</td><td>60.4</td><td>84.2</td><td>92.9</td><td>80.2</td><td>96.7</td><td>99.2</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14@336px</sub></td><td>64.7</td><td>89.6</td><td>94.6</td><td>80.1</td><td>96.7</td><td><b>99.2</b></td><td><b>63.4</b></td><td><b>87.2</b></td><td><b>94.4</b></td><td>81.2</td><td>97.2</td><td>99.1</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-H/14</sub></td><td><b>69.2</b></td><td><b>89.9</b></td><td><b>96.1</b></td><td><b>81.5</b></td><td><b>96.9</b></td><td>99.1</td><td>63.0</td><td>86.6</td><td>92.9</td><td><b>83.5</b></td><td><b>97.3</b></td><td><b>99.2</b></td>
    </tr>  
</table>
<br>

**Zero-shot Image Classification**:<span id="zeroshot_results"></span>
<table border="1" width="150%">
	<tr align="center">
        <th>Task</th><th>CIFAR10</th><th>CIFAR100</th><th>DTD</th><th>EuroSAT</th><th>FER</th><th>FGVC</th><th>KITTI</th><th>MNIST</th><th>PC</th><th>VOC</th><th>ImageNet</th>
    </tr>
	<tr align="center">
        <td width="150%">CN-CLIP<sub>RN50</sub></td><td>72.7</td><td>40.6</td><td>36.9</td><td>27.0</td><td>21.9</td><td>5.4</td><td>30.2</td><td>50.2 </td><td>47.7</td><td>82.1</td><td>33.5</td>
    </tr>
    	<tr align="center">
        <td width="150%">CN-CLIP<sub>ViT-B/16</sub></td><td>92.0</td><td>64.4</td><td>43.6</td><td>46.9</td><td>47.2</td><td>12.8</td><td>33.5</td><td>67.6 </td><td>54.0</td><td>83.3</td><td>48.3</td>
    </tr>
	<tr align="center">
        <td width="150%">CN-CLIP<sub>ViT-L/14</sub></td><td>94.9</td><td>75.1</td><td>44.2</td><td>56.9</td><td>54.6</td><td>16.0</td><td>49.9</td><td>69.8 </td><td>63.5</td><td>84.5</td><td>54.7</td>
    </tr>
    	<tr align="center">
        <td width="150%">CN-CLIP<sub>ViT-L/14@336px</sub></td><td>94.1</td><td>73.5</td><td>43.8</td><td>50.7</td><td>55.1</td><td>17.1</td><td>49.8</td><td>65.0</td><td>62.9</td><td>84.9</td><td>56.7</td>
    </tr>
    	<tr align="center">
        <td width="150%">CN-CLIP<sub>ViT-H/14</sub></td><td>96.0</td><td>79.7</td><td>51.2</td><td>52.0</td><td>49.2</td><td>26.2</td><td>39.1</td><td>79.4</td><td>52.4</td><td>84.9</td><td>59.6</td>
    </tr>
</table>
<br><br>
