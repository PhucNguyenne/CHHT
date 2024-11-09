document.addEventListener("DOMContentLoaded", function() {
    const mapDiv = document.getElementById("map");
    const toggleButton = document.getElementById("toggleMap");

    toggleButton.addEventListener("click", () => {
        if (mapDiv.style.pointerEvents === "none") {
            mapDiv.style.pointerEvents = "auto";
            toggleButton.innerText = "Tắt Tương Tác Bản Đồ";
        } else {
            mapDiv.style.pointerEvents = "none";
            toggleButton.innerText = "Bật Tương Tác Bản Đồ";
        }
    });
});
