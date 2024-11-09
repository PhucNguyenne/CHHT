// Hàm mở modal
function openModal() {
    document.getElementById("sevendays").style.display = "block";
}

// Hàm đóng modal
function closeModal() {
    document.getElementById("sevendays").style.display = "none";
}

// Đóng modal khi nhấn bên ngoài modal
window.onclick = function(event) {
    var modal = document.getElementById("sevendays");
    if (event.target == modal) {
        modal.style.display = "none";
    }
}
