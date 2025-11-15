import axios from 'axios';
const BACKEND = process.env.REACT_APP_BACKEND || "http://localhost:5000";

export async function fetchProducts(){
  const res = await axios.get(`${BACKEND}/products`);
  return res.data;
}

export async function searchImageFile(file, k=12){
  const form = new FormData();
  form.append("image", file);
  form.append("k", k);
  const res = await axios.post(`${BACKEND}/search`, form, { headers: {'Content-Type': 'multipart/form-data'} });
  return res.data;
}

export async function searchImageUrl(image_url, k=12){
  const form = new FormData();
  form.append("image_url", image_url);
  form.append("k", k);
  const res = await axios.post(`${BACKEND}/search`, form);
  return res.data;
}
