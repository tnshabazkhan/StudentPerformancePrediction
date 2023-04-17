function showAttributesUsed()
{
    document.getElementById('dataVisualisation').style.display = "none";
    document.getElementById('featureExtraction').style.display = "none";
    document.getElementById('algorithmsUsed').style.display = "none";
    document.getElementById('attributesUsed').style.display = "block";
    document.getElementById('prediction').style.display = "none";


}

function showDataVisualisation()
{
    document.getElementById('dataVisualisation').style.display = "block";
    document.getElementById('featureExtraction').style.display = "none";
    document.getElementById('algorithmsUsed').style.display = "none";
    document.getElementById('attributesUsed').style.display = "none";
    document.getElementById('prediction').style.display = "none";


}
function  showFeatureExtraction()
{
    document.getElementById('dataVisualisation').style.display = "none";
    document.getElementById('featureExtraction').style.display = "block";
    document.getElementById('algorithmsUsed').style.display = "none";
    document.getElementById('attributesUsed').style.display = "none";
    document.getElementById('prediction').style.display = "none";

}
function  showAlgorithmsUsed()
{
    document.getElementById('dataVisualisation').style.display = "none";
    document.getElementById('featureExtraction').style.display = "none";
    document.getElementById('algorithmsUsed').style.display = "block";
    document.getElementById('attributesUsed').style.display = "none";
    document.getElementById('prediction').style.display = "none";

}
function showPrediction()
{
    document.getElementById('dataVisualisation').style.display = "none";
    document.getElementById('featureExtraction').style.display = "none";
    document.getElementById('algorithmsUsed').style.display = "none";
    document.getElementById('attributesUsed').style.display = "none";
    document.getElementById('prediction').style.display = "block";
}
function myFunction() {
    document.getElementById("frm1").submit();
  }