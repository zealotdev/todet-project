$(document).ready(function() {
  $('.image-preview').magnificPopup({type:'image'});

});

window.addEventListener("load", function(){
  const loader = document.querySelector(".loader");
  loader.className += " hidden";
});
