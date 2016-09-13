

__device__
void vector3_add(float3* a, const float3* b)
{
  a->x += b->x;
  a->y += b->y;
  a->z += b->z;
}

__device__
void vector3_sub(float3* a, const float3* b)
{
  a->x -= b->x;
  a->y -= b->y;
  a->z -= b->z;
}

__device__
float vector3_dot(const float3* a, const float3* b)
{
  float result = 0.f;
  
  result += a->x * b->x;
  result += a->y * b->y;
  result += a->z * b->z;
  
  return result;
}

__device__
void vector3_scale_add(float3* a, float s, const float3* b)
{
  a->x += s * b->x;
  a->y += s * b->y;
  a->z += s * b->z;
}

__device__
void vector3_scale(float3* a, float s)
{
  a->x *= s;
  a->y *= s;
  a->z *= s;
}

__device__
void vector3_axpy(float3* a, float s, const float3* b)
{
  float3 temp = *b;

  temp.x += a->x * s;
  temp.y += a->y * s;
  temp.z += a->z * s;

  *a = temp;
}

__device__
void vector3_crossp(const float3* a, const float3* b, float3* out)
{
  out->x =  a->y * b->z;
  out->x -= a->z * b->y;
  
  out->y =  a->z * b->x;
  out->y -= a->x * b->z;
  
  out->z =  a->x * b->y;
  out->z -= a->y * b->x;
}

struct matrix3x3
{
  float3 row0;
  float3 row1;
  float3 row2;
};

__device__
void matrix3x3_vector_mul(const matrix3x3* m, const float3* a, float* rhs)
{
  rhs->x = vector3_dot(&(m->row0), a);
  rhs->y = vector3_dot(&(m->row1), a);
  rhs->z = vector3_dot(&(m->row2), a);
}





